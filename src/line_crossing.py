import numpy as np
from collections import defaultdict
import cv2

class LineCounter:
    def __init__(self, line_start, line_end, line_offset=10):
        """
        Initialize line counter with a virtual line defined by two points.
        line_start: (x1, y1) tuple - top of line
        line_end: (x2, y2) tuple - bottom of line
        line_offset: offset distance to create counting zones
        """
        self.line_start = np.array(line_start)
        self.line_end = np.array(line_end)
        self.line_vector = self.line_end - self.line_start
        self.line_length = np.linalg.norm(self.line_vector)
        self.line_unit_vector = self.line_vector / self.line_length
        self.normal_vector = np.array([-self.line_unit_vector[1], self.line_unit_vector[0]])
        
        # Create counting zones with asymmetric offsets
        self.center_offset_left = line_offset  # Full width on door side
        self.center_offset_right = int(line_offset * .85)  # Narrower on left enterance
        self.left_offset = int(line_offset * 1)  # Right exit zone
        self.right_offset = line_offset  # Left exit zone
        self.zone_points = self._create_counting_zones()
        
        # Counting stats
        self.counts = {"in": 0, "out_left": 0, "out_right": 0}
        self.tracked_crossings = defaultdict(lambda: {"last_pos": None, "zone": None, "counted": False})
        
    def _create_counting_zones(self):
        """Create polygons for entry/exit zones"""
        # Center zone (asymmetric)
        offset_left = -self.normal_vector * self.center_offset_left
        offset_right = self.normal_vector * self.center_offset_right
        
        # Exit zones
        far_left = -self.normal_vector * (self.center_offset_left + self.left_offset)
        far_right = self.normal_vector * (self.center_offset_right + self.right_offset)
        
        return {
            "center": np.array([
                self.line_start + offset_left,
                self.line_end + offset_left,
                self.line_end + offset_right,
                self.line_start + offset_right
            ], dtype=np.int32),
            "left": np.array([
                self.line_start + far_left,
                self.line_end + far_left,
                self.line_end + offset_left,
                self.line_start + offset_left
            ], dtype=np.int32),
            "right": np.array([
                self.line_start + offset_right,
                self.line_end + offset_right,
                self.line_end + far_right,
                self.line_start + far_right
            ], dtype=np.int32)
        }
    
    def _point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon"""
        return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0
    
    def update(self, tracks):
        """
        Update counting based on current tracks
        tracks: dict of track_id -> {bbox: (x, y, w, h), ...}
        Returns: Updated counts dictionary
        """
        for track_id, track_info in tracks.items():
            try:
                bbox = track_info['bbox']
                # Get center point of bbox bottom
                current_pos = (int(bbox[0] + bbox[2]//2), int(bbox[1] + bbox[3]))  # Convert to tuple of ints
                
                track_data = self.tracked_crossings[track_id]
                last_pos = track_data["last_pos"]
                current_zone = None
                
                # Determine current zone
                if self._point_in_polygon(current_pos, self.zone_points["center"]):
                    current_zone = "center"
                elif self._point_in_polygon(current_pos, self.zone_points["left"]):
                    current_zone = "left"
                elif self._point_in_polygon(current_pos, self.zone_points["right"]):
                    current_zone = "right"
                
                if last_pos is not None and not track_data["counted"]:
                    # Check for zone transitions
                    if track_data["zone"] == "center":
                        if current_zone == "left":
                            self.counts["out_left"] += 1
                            track_data["counted"] = True
                        elif current_zone == "right":
                            self.counts["out_right"] += 1
                            track_data["counted"] = True
                    elif (track_data["zone"] == "left" or track_data["zone"] == "right") and current_zone == "center":
                        self.counts["in"] += 1
                        track_data["counted"] = True
                
                # Update tracking data
                track_data["last_pos"] = current_pos
                track_data["zone"] = current_zone
                
                # Reset counted flag if person moves back to original zone
                if current_zone != track_data["zone"]:
                    track_data["counted"] = False
                    
            except Exception as e:
                print(f"Error processing track {track_id}: {str(e)}")
                continue
        
        return self.counts
    
    def draw(self, frame):
        """Draw counting line and zones on frame"""
        # Draw center line
        cv2.line(frame, tuple(self.line_start), tuple(self.line_end), (0, 255, 0), 2)
        
        # Draw zones with transparency
        overlay = frame.copy()
        # Center zone (green)
        cv2.fillPoly(overlay, [self.zone_points["center"]], (0, 255, 0, 128))
        # Left exit zone (red)
        cv2.fillPoly(overlay, [self.zone_points["left"]], (0, 0, 255, 128))
        # Right exit zone (red)
        cv2.fillPoly(overlay, [self.zone_points["right"]], (0, 0, 255, 128))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw counts
        total_out = self.counts["out_left"] + self.counts["out_right"]
        cv2.putText(frame, f"IN: {self.counts['in']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"OUT: {total_out}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return frame 