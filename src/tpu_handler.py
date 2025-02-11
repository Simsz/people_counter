import logging
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
import threading
import queue
import time
from PIL import Image, ImageOps

class TPUHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.delegate = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.initialize_tpu()

    def initialize_tpu(self):
        try:
            # Initialize Edge TPU
            self.delegate = edgetpu.load_edgetpu_delegate()
            self.interpreter = edgetpu.make_interpreter(self.model_path, delegate=self.delegate)
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logging.info(f"TPU initialized successfully")
            logging.debug(f"Input details: {self.input_details}")
            logging.debug(f"Output details: {self.output_details}")
            
        except Exception as e:
            logging.error(f"Failed to initialize TPU: {str(e)}")
            raise

    def process_frame(self, frame, threshold=0.5):
        if self.interpreter is None:
            raise RuntimeError("TPU not initialized")

        # Convert frame to PIL Image
        image = Image.fromarray(frame)
        
        # Prepare input
        _, height, width, _ = self.input_details[0]['shape']
        image = image.resize((width, height), Image.LANCZOS)
        
        # Convert to numpy array and ensure UINT8 type
        input_data = np.array(image, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Set the tensor directly without normalization
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        count = int(self.interpreter.get_tensor(self.output_details[3]['index']))

        detections = []
        for i in range(count):
            if scores[0][i] >= threshold:
                box = boxes[0][i]
                detections.append({
                    'bbox': [
                        int(box[1] * frame.shape[1]),  # xmin
                        int(box[0] * frame.shape[0]),  # ymin
                        int(box[3] * frame.shape[1]),  # xmax
                        int(box[2] * frame.shape[0])   # ymax
                    ],
                    'class': int(classes[0][i]),
                    'confidence': float(scores[0][i])
                })

        return detections 