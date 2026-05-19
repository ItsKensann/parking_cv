"""
detector.py
-----------
Core car detection and counting pipeline using YOLOv8.
Handles both occupancy counting (lite tier) and spot-level detection (pro tier).
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

# YOLO class index for 'car' in the COCO dataset
# Also includes truck (7), bus (5) if you want to count all vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class ParkingSpot:
    """Represents a single defined parking spot (pro tier)."""
    spot_id: str
    polygon: np.ndarray          # Shape: (N, 2) array of (x, y) points
    is_occupied: bool = False
    status: str = "unknown"
    confidence: float = 0.0
    evidence_history: deque = field(default_factory=deque, repr=False)
    _last_stable_status: str = "unknown"
    _ambiguous_frames: int = 0


@dataclass(frozen=True)
class VehicleDetection:
    """Vehicle box and detector confidence for a single frame."""
    box: tuple[int, int, int, int]
    confidence: float
    is_split: bool = False


@dataclass
class DetectionResult:
    """Output from a single frame of inference."""
    frame: np.ndarray            # Annotated frame for display/streaming
    car_count: int               # Number of cars detected in the frame
    smoothed_count: int          # Smoothed count to reduce flickering
    occupancy_pct: float         # 0.0 - 1.0, requires capacity to be set
    spots: list[ParkingSpot] = field(default_factory=list)  # Pro tier only


class ParkingDetector:
    """
    Detects cars in video frames and determines parking occupancy.

    Usage (lite tier - just counting):
        detector = ParkingDetector(capacity=50)
        result = detector.process_frame(frame)
        print(result.occupancy_pct)

    Usage (pro tier - spot level):
        detector = ParkingDetector(capacity=50)
        detector.load_spots("spots.json")
        result = detector.process_frame(frame)
        for spot in result.spots:
            print(spot.spot_id, spot.is_occupied)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",   # nano = fastest and most accurate
        capacity: int = 100,               # Total number of spots in the lot
        confidence_threshold: float = 0.14,
        smoothing_window: int = 30,         # Frames to average for count smoothing
        iou_threshold: float = 0.15,        # Overlap needed to mark a spot occupied
        spot_smoothing_window: int = 10,
        spot_stable_frames: int = 3,
        spot_occupied_frames: int = 4,
        spot_hold_frames: int = 4,
        spot_overlap_threshold: float = 0.36,
        spot_strong_overlap_threshold: float = 0.55,
        spot_overlap_high_confidence: float = 0.65,
        spot_occupied_score_threshold: float = 0.72,
        spot_clear_score_threshold: float = 0.15,
    ):
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.capacity = capacity
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.spot_smoothing_window = spot_smoothing_window
        self.spot_stable_frames = spot_stable_frames
        self.spot_occupied_frames = spot_occupied_frames
        self.spot_hold_frames = spot_hold_frames
        self.spot_overlap_threshold = spot_overlap_threshold
        self.spot_strong_overlap_threshold = spot_strong_overlap_threshold
        self.spot_overlap_high_confidence = spot_overlap_high_confidence
        self.spot_occupied_score_threshold = spot_occupied_score_threshold
        self.spot_clear_score_threshold = spot_clear_score_threshold

        # Rolling window for count smoothing (prevents flickering)
        self._count_history: deque = deque(maxlen=smoothing_window)

        # Pro tier: defined parking spots
        self.spots: list[ParkingSpot] = []

    # ------------------------------------------------------------------
    # Spot management (pro tier)
    # ------------------------------------------------------------------

    def load_spots(self, spots_config: list[dict]):
        """
        Load parking spot definitions from a list of dicts.
        Each dict: { "id": "A1", "polygon": [[x1,y1], [x2,y2], ...] }
        """
        self.spots = [
            ParkingSpot(
                spot_id=s["id"],
                polygon=np.array(s["polygon"], dtype=np.int32),
                evidence_history=deque(maxlen=self.spot_smoothing_window),
            )
            for s in spots_config
        ]
        print(f"Loaded {len(self.spots)} parking spots")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a single frame and return detection results."""
        frame = self._preprocess_frame(frame)
        results = self.model(frame, verbose=False, imgsz=1280)[0]

        # Filter to vehicle classes above confidence threshold
        vehicle_detections: list[VehicleDetection] = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in VEHICLE_CLASSES and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_detections.append(VehicleDetection((x1, y1, x2, y2), conf))

        # Smooth the count
        self._count_history.append(len(vehicle_detections))
        smoothed = int(round(sum(self._count_history) / len(self._count_history)))

        # Occupancy percentage
        occupancy_pct = min(smoothed / self.capacity, 1.0) if self.capacity > 0 else 0.0

        vehicle_detections = self._split_merged_boxes(vehicle_detections)
        if self.spots:
            self._update_spot_occupancy(vehicle_detections)

        # Draw annotations on frame
        car_boxes = [detection.box for detection in vehicle_detections]
        annotated = self._draw_annotations(frame.copy(), car_boxes, smoothed, occupancy_pct)

        return DetectionResult(
            frame=annotated,
            car_count=len(vehicle_detections),
            smoothed_count=smoothed,
            occupancy_pct=occupancy_pct,
            spots=list(self.spots),
        )

    # ------------------------------------------------------------------
    # Spot occupancy check via anchor point, polygon overlap, and smoothing
    # ------------------------------------------------------------------

    # helper function for the case that two boxes get merged
    def _split_merged_boxes(
        self,
        vehicle_detections: list[VehicleDetection],
        width_ratio: float = 1.8,
    ) -> list[VehicleDetection]:
        """
        Splits bounding boxes that are suspiciously wide into two equal halves.
        These are likely two cars that YOLO merged into one detection.
        """
        result = []
        for detection in vehicle_detections:
            x1, y1, x2, y2 = detection.box
            w = x2 - x1
            h = y2 - y1
            if h > 0 and (w / h) > width_ratio:
                mid = (x1 + x2) // 2
                split_confidence = detection.confidence * 0.75
                result.append(VehicleDetection((x1, y1, mid, y2), split_confidence, True))
                result.append(VehicleDetection((mid, y1, x2, y2), split_confidence, True))
            else:
                result.append(detection)
        return result

    def _update_spot_occupancy(self, vehicle_detections: list[VehicleDetection]):
        for spot in self.spots:
            evidence_score = self._score_spot_occupancy(spot, vehicle_detections)
            self._apply_spot_smoothing(spot, evidence_score)

    def _score_spot_occupancy(
        self,
        spot: ParkingSpot,
        vehicle_detections: list[VehicleDetection],
    ) -> float:
        best_score = 0.0
        anchor_inside_spot = False
        overlapping_detections = 0
        for detection in vehicle_detections:
            x1, y1, x2, y2 = detection.box
            confidence = detection.confidence
            spot_center_x = float(spot.polygon[:, 0].mean())
            spot_center_y = float(spot.polygon[:, 1].mean())
            box_contains_spot_center = (
                x1 <= spot_center_x <= x2 and y1 <= spot_center_y <= y2
            )

            anchor_x = (x1 + x2) // 2
            anchor_y = int(y1 + (y2 - y1) * 0.85)
            anchor_result = cv2.pointPolygonTest(
                spot.polygon, (float(anchor_x), float(anchor_y)), False
            )
            if anchor_result >= 0:
                anchor_inside_spot = True
                if detection.is_split:
                    anchor_score = 0.55 + (0.30 * confidence)
                else:
                    anchor_score = 0.62 + (0.35 * confidence)
                best_score = max(best_score, anchor_score)

            overlap = self._spot_coverage_by_box(spot.polygon, detection.box)
            if overlap >= self.spot_overlap_threshold:
                overlapping_detections += 1
                overlap_scale = min(
                    (overlap - self.spot_overlap_threshold)
                    / max(1.0 - self.spot_overlap_threshold, 0.01),
                    1.0,
                )
                strong_overlap = overlap >= self.spot_strong_overlap_threshold
                high_confidence = confidence >= self.spot_overlap_high_confidence

                if box_contains_spot_center or (strong_overlap and high_confidence):
                    overlap_score = 0.50 + (0.30 * overlap_scale) + (0.20 * confidence)
                    if not box_contains_spot_center:
                        overlap_score -= 0.14
                    if detection.is_split:
                        overlap_score -= 0.10
                    best_score = max(best_score, overlap_score)
                else:
                    # Weak edge overlap usually means a neighboring car box bled over the stall line.
                    best_score = max(best_score, min(0.35, overlap_scale * confidence))

        # Several neighboring boxes can cover an empty stall without any car owning it.
        if not anchor_inside_spot and overlapping_detections >= 2:
            return min(best_score, self.spot_occupied_score_threshold - 0.05)

        return min(max(best_score, 0.0), 1.0)

    def _apply_spot_smoothing(self, spot: ParkingSpot, evidence_score: float):
        spot.evidence_history.append(evidence_score)
        evidence = list(spot.evidence_history)
        available_recent = evidence[-self.spot_stable_frames:]
        occupied_recent = evidence[-self.spot_occupied_frames:]

        if len(available_recent) < self.spot_stable_frames:
            self._set_spot_status(spot, "unknown", 0.0)
            return

        if (
            len(occupied_recent) >= self.spot_occupied_frames
            and all(score >= self.spot_occupied_score_threshold for score in occupied_recent)
        ):
            confidence = sum(occupied_recent) / len(occupied_recent)
            self._set_spot_status(spot, "occupied", confidence, stable=True)
            return

        if all(score <= self.spot_clear_score_threshold for score in available_recent):
            confidence = sum(1.0 - score for score in available_recent) / len(available_recent)
            self._set_spot_status(spot, "available", confidence, stable=True)
            return

        spot._ambiguous_frames += 1
        if spot._last_stable_status != "unknown" and spot._ambiguous_frames <= self.spot_hold_frames:
            decayed_confidence = max(0.35, spot.confidence * 0.85)
            self._set_spot_status(spot, spot._last_stable_status, decayed_confidence)
            return

        self._set_spot_status(spot, "unknown", 0.0)

    @staticmethod
    def _set_spot_status(
        spot: ParkingSpot,
        status: str,
        confidence: float,
        stable: bool = False,
    ):
        spot.status = status
        spot.is_occupied = status == "occupied"
        spot.confidence = round(min(max(confidence, 0.0), 1.0), 3)
        if stable:
            spot._last_stable_status = status
            spot._ambiguous_frames = 0

    @staticmethod
    def _spot_coverage_by_box(polygon: np.ndarray, box: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return 0.0

        poly_x, poly_y, poly_w, poly_h = cv2.boundingRect(polygon)
        roi_x1 = max(x1, poly_x)
        roi_y1 = max(y1, poly_y)
        roi_x2 = min(x2, poly_x + poly_w)
        roi_y2 = min(y2, poly_y + poly_h)
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return 0.0

        width = roi_x2 - roi_x1
        height = roi_y2 - roi_y1
        shifted_polygon = polygon - np.array([roi_x1, roi_y1], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_polygon], 255)

        intersection_area = float(cv2.countNonZero(mask))
        spot_area = max(float(cv2.contourArea(polygon)), 1.0)
        return min(intersection_area / spot_area, 1.0)

    @staticmethod
    def _compute_iou(box_a: tuple, box_b: tuple) -> float:
        """
        Compute Intersection over Union between two boxes.
        Each box: (x1, y1, x2, y2)
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0:
            return 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _draw_annotations(
        self,
        frame: np.ndarray,
        car_boxes: list[tuple],
        smoothed_count: int,
        occupancy_pct: float,
    ) -> np.ndarray:
        """Draw bounding boxes, spot overlays, and HUD onto the frame."""

        # Draw car bounding boxes
        for (x1, y1, x2, y2) in car_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw parking spot polygons (pro tier)
        for spot in self.spots:
            if spot.status == "occupied":
                color = (0, 0, 255)      # red
            elif spot.status == "unknown":
                color = (0, 255, 255)    # yellow
            else:
                color = (0, 255, 0)      # green
            cv2.polylines(frame, [spot.polygon], isClosed=True, color=color, thickness=2)
            cx = int(spot.polygon[:, 0].mean())
            cy = int(spot.polygon[:, 1].mean())
            cv2.putText(frame, spot.spot_id, (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # HUD overlay
        occupied = smoothed_count
        available = max(self.capacity - occupied, 0)
        pct_label = f"{occupancy_pct * 100:.0f}%"

        # Color the HUD based on occupancy
        if occupancy_pct < 0.6:
            hud_color = (0, 200, 0)    # green
        elif occupancy_pct < 0.85:
            hud_color = (0, 165, 255)  # orange
        else:
            hud_color = (0, 0, 255)    # red

        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)  # black background
        cv2.putText(frame, f"Vehicles detected: {occupied}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Available spots:   {available}/{self.capacity}", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Occupancy: {pct_label}", (20, 81),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

        return frame
