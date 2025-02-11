import logging
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
import threading
import queue
import time
from PIL import Image, ImageOps
import cv2
import os

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

        # Get original frame dimensions
        original_height, original_width = frame.shape[:2]
        
        # Get model input shape (should be 300x300 for this model)
        input_shape = self.input_details[0]['shape']
        model_height, model_width = input_shape[1], input_shape[2]
        
        # Resize frame to model input size
        frame_resized = cv2.resize(frame, (model_width, model_height))
        
        # Convert frame to PIL Image
        image = Image.fromarray(frame_resized)
        
        # Convert to numpy array and ensure UINT8 type
        input_data = np.array(image, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Set the tensor
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
                # Scale coordinates back to original frame size
                detections.append({
                    'bbox': [
                        max(0, int(box[1] * original_width)),   # xmin
                        max(0, int(box[0] * original_height)),  # ymin
                        min(original_width, int(box[3] * original_width)),   # xmax
                        min(original_height, int(box[2] * original_height))  # ymax
                    ],
                    'class': int(classes[0][i]),
                    'confidence': float(scores[0][i])
                })
                # Debug print
                print(f"Detection {i}: bbox={detections[-1]['bbox']}, conf={scores[0][i]:.2f}")

        return detections 