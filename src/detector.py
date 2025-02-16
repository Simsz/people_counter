import cv2
import numpy as np
from .tpu_handler import TPUHandler
from .tracker import PersonTracker
import logging

# We'll add person detection logic here later
class PersonDetector:
    def __init__(self, model_path):
        self.tpu = TPUHandler(model_path)
        self.confidence_threshold = 0.30  # Lowered for better detection in darker areas
        self.person_class_id = 0
        self.tracker = PersonTracker(max_disappeared=20, min_confidence=0.6)
        
        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=20, detectShadows=False)
        self.min_motion_area = 800  # Increased for full-body detection
        self.last_frame = None
        
        # NMS parameters
        self.nms_threshold = 0.3
        print(f"Initialized detector with model: {model_path}")
        
    def _detect_motion(self, frame):
        """Detect areas of motion in the frame"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_motion_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, x + w, y + h))
        
        return motion_regions
        
    def _bbox_overlap(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
        
    def _apply_nms(self, detections):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
            
        # Convert to numpy arrays for easier processing
        boxes = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by confidence
        idxs = np.argsort(scores)
        
        # Initialize the list of picked indexes
        pick = []
        
        while len(idxs) > 0:
            # Pick the last index and add it to the list
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find the intersection
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[:last], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[:last], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[:last], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[:last], 3])
            
            # Calculate intersection area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            # Calculate IoU
            union = areas[i] + areas[idxs[:last]] - intersection
            overlap = intersection / (union + 1e-6)
            
            # Delete all indexes from the index list that have high overlap
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.nms_threshold)[0])))
        
        return [detections[i] for i in pick]
        
    def detect(self, frame):
        try:
            # Detect motion regions
            motion_regions = self._detect_motion(frame)
            
            # Get detections from TPU
            detections = self.tpu.process_frame(frame, threshold=self.confidence_threshold)
            
            # Filter for person class and validate against motion
            person_detections = []
            for d in detections:
                if d['class'] == self.person_class_id:
                    bbox = [int(x) for x in d['bbox']]
                    
                    # Check if detection overlaps with any motion region
                    motion_validated = False
                    for motion_bbox in motion_regions:
                        if self._bbox_overlap(bbox, motion_bbox) > 0.3:
                            motion_validated = True
                            break
                    
                    # Accept detection if it's high confidence or motion validated
                    if d['confidence'] > 0.6 or motion_validated:
                        d['bbox'] = bbox
                        person_detections.append(d)
            
            # Apply NMS to remove overlapping detections
            person_detections = self._apply_nms(person_detections)
            
            # Update tracks
            tracks = self.tracker.update(frame, person_detections)
            return person_detections, tracks, motion_regions
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], {}, []
        
    def draw_detections(self, frame, detection_data):
        try:
            detections, tracks, motion_regions = detection_data
            height, width = frame.shape[:2]
            
            # Draw motion regions
            for region in motion_regions:
                x1, y1, x2, y2 = region
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
            
            # Draw detections
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
                    
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
                    label = f"{confidence:.2f}"
                    cv2.putText(frame, label, (xmin, ymin-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                except Exception as e:
                    print(f"Error drawing detection: {str(e)}")
                    continue
            
            # Draw tracked objects
            for track_id, track in tracks.items():
                try:
                    x, y, w, h = [int(v) for v in track['bbox']]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, str(track_id), (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error drawing track {track_id}: {str(e)}")
                    continue
            
            # Add stats overlay
            stats = f"Tracked: {len(tracks)} | Detections: {len(detections)}"
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, stats, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Drawing error: {str(e)}")
            return frame 