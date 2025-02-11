import cv2
from .tpu_handler import TPUHandler
from .tracker import PersonTracker
import logging

# We'll add person detection logic here later
class PersonDetector:
    def __init__(self, model_path):
        self.tpu = TPUHandler(model_path)
        self.confidence_threshold = 0.35
        self.person_class_id = 0  # Changed back to 0 for this model
        self.tracker = PersonTracker(max_disappeared=30, min_confidence=0.7)
        print(f"Initialized detector with model: {model_path}")
        
    def detect(self, frame):
        try:
            detections = self.tpu.process_frame(frame, threshold=self.confidence_threshold)
            # Debug print
            print(f"Raw detections: {detections}")
            
            # Filter for person class only and ensure bbox format
            person_detections = []
            for d in detections:
                # Print class information for debugging
                print(f"Detection class: {d['class']}, confidence: {d['confidence']}")
                
                if d['class'] == self.person_class_id:
                    # Ensure bbox values are integers
                    d['bbox'] = [int(x) for x in d['bbox']]
                    person_detections.append(d)
            
            # Debug print
            print(f"Person detections: {person_detections}")
            
            # Update tracks
            tracks = self.tracker.update(frame, person_detections)
            return person_detections, tracks
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], {}
        
    def draw_detections(self, frame, detection_data):
        try:
            detections, tracks = detection_data
            height, width = frame.shape[:2]
            
            # First draw detections with confidence-based colors
            for det in detections:
                try:
                    xmin, ymin, xmax, ymax = det['bbox']
                    confidence = det['confidence']
                    
                    # Color based on confidence
                    if confidence > 0.6:
                        color = (0, 255, 0)  # Green
                    elif confidence > 0.5:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 165, 255)  # Orange
                    
                    # Draw thinner box for detections
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
                    
                    # Add small confidence label
                    label = f"{confidence:.2f}"
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.putText(frame, label, (xmin, ymin-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                except Exception as e:
                    print(f"Error drawing detection: {str(e)}")
                    continue
            
            # Then draw tracked objects with bold green boxes
            for track_id, track in tracks.items():
                try:
                    x, y, w, h = [int(v) for v in track['bbox']]
                    
                    # Draw bold green box for tracking
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                except Exception as e:
                    print(f"Error drawing track {track_id}: {str(e)}")
                    continue
            
            # Add stats overlay with semi-transparent background
            stats = f"Tracked Objects: {len(tracks)} | Detections: {len(detections)}"
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, stats, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Drawing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame 