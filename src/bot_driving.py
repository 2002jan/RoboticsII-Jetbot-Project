import cv2
import onnxruntime as rt

from pathlib import Path
import time
import yaml
import numpy as np

from PUTDriver import PUTDriver, gstreamer_pipeline


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        # averaging
        self.queue_length = 10  
        self.last_frames = [(0.0, 0.0) for _ in range(self.queue_length)]

    def preprocess(self, img: np.ndarray) -> np.ndarray:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        # make sure the output is between -1 and 1
        detections = np.clip(detections, -1.0, 1.0)
        detections = detections.reshape(2,)
        return detections

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        self.update_queue(outputs)
        averaged_outputs = self.exponential_average(alpha=1.1)


        assert outputs.max() <= 1.0
        assert outputs.min() >= -1.0

        return averaged_outputs
    
    def update_queue(self, new_prediction):
        self.last_frames.pop(0)
        self.last_frames.append(new_prediction)

    def exponential_average(self, alpha=1.2):
        weights = np.array([alpha**i for i in range(self.queue_length)])
        weights /= weights.sum()
        weighted_predictions = (np.array(self.last_frames) * weights[:, np.newaxis]).sum(axis=0)

        return weighted_predictions.astype(np.float32)
    




def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    # taken from https://github.com/JanekDev/RoboticsII-Jetbot/blob/dev/src/bot_driving.py
    WARMUP_FRAMES = 30
    print("Warming up...")
    warmup_tic = time.time()
    for _n in range(WARMUP_FRAMES):
        ret, image = video_capture.read() 
        if not ret:
            print('No camera')
            return
        forward, left = ai.predict(image)
    warmup_tac = time.time()
    print(f" [ok] Took {warmup_tac - warmup_tic} seconds "
          f"({((warmup_tac - warmup_tic) * 1000 / WARMUP_FRAMES):.3f} ms per frame)")

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break
        forward, left = ai.predict(image)


if __name__ == '__main__':
    main()
