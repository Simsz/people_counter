import cv2
import numpy as np
from datetime import datetime, timedelta

class PersonTracker:
    def __init__(self, max_disappeared=30, min_confidence=0.7):
        self.next_track_id = 0
        self.tracks = {}  # {track_id: {tracker, bbox, last_confidence, last_seen, disappeared_count}}
        self.max_disappeared = max_disappeared
        self.min_confidence = min_confidence
        
    def _create_tracker(self):
        # You can experiment with different trackers:
        # CSRT - More accurate but slower
        # KCF - Faster but less accurate
        # MOSSE - Fastest but less accurate
        return cv2.TrackerKCF_create()
        
    def update(self, frame, detections):
        try:
            current_tracks = set()
            
            # Update existing tracks
            for track_id in list(self.tracks.keys()):
                try:
                    track = self.tracks[track_id]
                    success, bbox = track['tracker'].update(frame)
                    
                    if success:
                        # Ensure bbox values are integers
                        track['bbox'] = tuple(int(x) for x in bbox)
                        track['disappeared_count'] = 0
                        current_tracks.add(track_id)
                    else:
                        track['disappeared_count'] += 1
                        if track['disappeared_count'] > self.max_disappeared:
                            del self.tracks[track_id]
                            continue
                        current_tracks.add(track_id)
                except Exception as e:
                    print(f"Error updating track {track_id}: {str(e)}")
                    continue
            
            # Process new detections
            for det in detections:
                try:
                    if det['confidence'] >= self.min_confidence:
                        # Get bbox coordinates
                        bbox = det['bbox']
                        # Convert [xmin, ymin, xmax, ymax] to [x, y, w, h]
                        x = int(bbox[0])
                        y = int(bbox[1])
                        w = int(bbox[2] - bbox[0])
                        h = int(bbox[3] - bbox[1])
                        tracker_bbox = (x, y, w, h)
                        
                        matched = False
                        
                        # Check if detection overlaps with existing tracks
                        for track_id in current_tracks:
                            track_bbox = self.tracks[track_id]['bbox']
                            if self._calculate_iou(tracker_bbox, track_bbox) > 0.3:
                                tracker = self._create_tracker()
                                tracker.init(frame, tracker_bbox)
                                self.tracks[track_id].update({
                                    'tracker': tracker,
                                    'bbox': tracker_bbox,
                                    'last_confidence': det['confidence'],
                                    'last_seen': datetime.now()
                                })
                                matched = True
                                break
                        
                        if not matched:
                            tracker = self._create_tracker()
                            tracker.init(frame, tracker_bbox)
                            self.tracks[self.next_track_id] = {
                                'tracker': tracker,
                                'bbox': tracker_bbox,
                                'last_confidence': det['confidence'],
                                'last_seen': datetime.now(),
                                'disappeared_count': 0
                            }
                            self.next_track_id += 1
                except Exception as e:
                    print(f"Error processing detection: {str(e)}")
                    continue
            
            return self.tracks
            
        except Exception as e:
            print(f"Tracker update error: {str(e)}")
            return {}
    
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