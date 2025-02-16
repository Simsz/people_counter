import cv2
import numpy as np
from .tpu_handler import TPUHandler
from .tracker import PersonTracker
import logging

# We'll add person detection logic here later
class PersonDetector:
    def __init__(self, model_path):
        self.tpu = TPUHandler(model_path)
        self.confidence_threshold = 0.30
        self.person_class_id = 0
        self.tracker = PersonTracker(max_disappeared=20, min_confidence=0.6)
        
        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False)  # Reduced history, increased threshold
        self.min_motion_area = 600  # Reduced minimum area
        self.last_frame = None
        
        # NMS parameters
        self.nms_threshold = 0.45  # Increased to reduce NMS processing
        self.detection_interval = 3  # Only run detection every N frames
        self.frame_count = 0
        self.last_detections = []
        print(f"Initialized detector with model: {model_path}")
        
    def _detect_motion(self, frame):
        """Detect areas of motion in the frame"""
        # Downscale frame for motion detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(small_frame)
        
        # Remove shadows and noise in one step
        _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        # Find contours (simpler method)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and convert back to original scale
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (self.min_motion_area / 4):  # Adjusted for downscaled image
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x*2, y*2, (x+w)*2, (y+h)*2))
        
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
            self.frame_count += 1
            motion_regions = self._detect_motion(frame)
            
            # Only run TPU detection every N frames or if there's significant motion
            if self.frame_count % self.detection_interval == 0 or len(motion_regions) > 0:
                # Get detections from TPU
                detections = self.tpu.process_frame(frame, threshold=self.confidence_threshold)
                
                # Filter for person class and validate against motion
                person_detections = []
                for d in detections:
                    if d['class'] == self.person_class_id:
                        bbox = [int(x) for x in d['bbox']]
                        
                        # Calculate aspect ratio and area
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        aspect_ratio = height / width if width > 0 else 0
                        area = width * height
                        
                        # Filter out unrealistic detections
                        if aspect_ratio < 1.0 or aspect_ratio > 4.0:  # Person should be taller than wide
                            continue
                        if area > (frame.shape[0] * frame.shape[1]) / 4:  # Too large (>25% of frame)
                            continue
                        if width < 20 or height < 40:  # Too small
                            continue
                        
                        # Accept detection if it's high confidence
                        if d['confidence'] > 0.6:
                            d['bbox'] = bbox
                            person_detections.append(d)
                        # Otherwise check motion validation
                        else:
                            for motion_bbox in motion_regions:
                                if self._bbox_overlap(bbox, motion_bbox) > 0.3:
                                    d['bbox'] = bbox
                                    person_detections.append(d)
                                    break
                
                # Apply stricter NMS
                self.last_detections = self._apply_nms(person_detections)
            
            # Update tracks with last known detections
            tracks = self.tracker.update(frame, self.last_detections)
            return self.last_detections, tracks, motion_regions
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], {}, []
            
    def draw_detections(self, frame, detection_data):
        try:
            detections, tracks, motion_regions = detection_data
            height, width = frame.shape[:2]
            
            # Draw only tracked objects (skip other visualizations if FPS is low)
            if self.frame_count % 2 == 0:  # Only draw every other frame
                # Draw motion regions
                for region in motion_regions:
                    x1, y1, x2, y2 = region
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                
                # Draw detections
                for det in detections:
                    xmin, ymin, xmax, ymax = det['bbox']
                    confidence = det['confidence']
                    color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            
            # Always draw tracked objects
            for track_id, track in tracks.items():
                try:
                    x, y, w, h = [int(v) for v in track['bbox']]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, str(track_id), (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1.5)
                except Exception as e:
                    continue
            
            # Add minimal stats overlay
            stats = f"Tracked: {len(tracks)}"
            cv2.putText(frame, stats, (10, 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"Drawing error: {str(e)}")
            return frame 