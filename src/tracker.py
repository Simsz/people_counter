import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

class PersonTracker:
    def __init__(self, max_disappeared=20, min_confidence=0.6):
        self.next_track_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.min_confidence = min_confidence
        self.min_iou_threshold = 0.3
        self.max_track_age = 30  # Maximum age of a track in frames
        
    def _create_tracker(self):
        # You can experiment with different trackers:
        # CSRT - More accurate but slower
        # KCF - Faster but less accurate
        # MOSSE - Fastest but less accurate
        return cv2.TrackerKCF_create()
        
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to x1y1x2y2 format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
        
    def _smooth_bbox(self, current_bbox, previous_bbox, alpha=0.7):
        """Apply exponential smoothing to bounding box"""
        if previous_bbox is None:
            return current_bbox
            
        x1, y1, w1, h1 = previous_bbox
        x2, y2, w2, h2 = current_bbox
        
        return [
            int(alpha * x1 + (1 - alpha) * x2),
            int(alpha * y1 + (1 - alpha) * y2),
            int(alpha * w1 + (1 - alpha) * w2),
            int(alpha * h1 + (1 - alpha) * h2)
        ]
        
    def update(self, frame, detections):
        # Convert detections to format [x, y, w, h]
        detection_bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            detection_bboxes.append([x1, y1, w, h])
        
        # Mark all existing tracks as unmatched
        unmatched_tracks = set(self.tracks.keys())
        matched_detections = set()
        
        # Match detections to existing tracks using IoU
        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            best_iou = self.min_iou_threshold
            best_detection = None
            best_idx = None
            
            for i, bbox in enumerate(detection_bboxes):
                if i in matched_detections:
                    continue
                    
                iou = self._calculate_iou(track['bbox'], bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = bbox
                    best_idx = i
            
            if best_detection is not None:
                # Update track with smoothed bbox
                smoothed_bbox = self._smooth_bbox(best_detection, track['bbox'])
                track.update({
                    'bbox': smoothed_bbox,
                    'confidence': detections[best_idx]['confidence'],
                    'disappeared': 0,
                    'age': track['age'] + 1
                })
                unmatched_tracks.remove(track_id)
                matched_detections.add(best_idx)
            else:
                track['disappeared'] += 1
        
        # Remove old tracks
        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            if track['disappeared'] > self.max_disappeared or track['age'] > self.max_track_age:
                del self.tracks[track_id]
        
        # Add new tracks for unmatched detections
        for i, bbox in enumerate(detection_bboxes):
            if i not in matched_detections and detections[i]['confidence'] >= self.min_confidence:
                self.tracks[self.next_track_id] = {
                    'bbox': bbox,
                    'confidence': detections[i]['confidence'],
                    'disappeared': 0,
                    'age': 0
                }
                self.next_track_id += 1
        
        return self.tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        b1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
        b2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
        
        # Calculate intersection
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0 