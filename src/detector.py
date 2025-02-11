import cv2
from .tpu_handler import TPUHandler

# We'll add person detection logic here later
class PersonDetector:
    def __init__(self, model_path):
        self.tpu = TPUHandler(model_path)
        
    def detect(self, frame):
        return self.tpu.process_frame(frame)
        
    def draw_detections(self, frame, detections):
        for det in detections:
            if det['class'] == 0:  # Assuming 0 is person class
                xmin, ymin, xmax, ymax = det['bbox']
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"Person {det['confidence']:.2f}"
                cv2.putText(frame, label, (xmin, ymin-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame 