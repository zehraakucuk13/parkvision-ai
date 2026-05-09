from pathlib import Path

import cv2
import numpy as np

from util import empty_or_not, get_parking_spots_bboxes


class ParkingDetector:
    def __init__(
        self,
        mask_path: str | Path,
        step: int = 30,
        change_threshold: float = 0.4,
        decision_box_scale: float = 0.65,
        roi: tuple[int, int, int, int] | None = None,
    ):
        self.mask_path = Path(mask_path)
        self.step = step
        self.change_threshold = change_threshold
        self.decision_box_scale = decision_box_scale
        self.roi = roi
        self.frame_number = 0
        self.previous_frame = None

        mask = cv2.imread(str(self.mask_path), 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {self.mask_path}")
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            mask = mask[y1:y2, x1:x2]

        connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        self.spots = get_parking_spots_bboxes(connected_components)
        self.spot_ids = [f"A{i + 1:02d}" for i in range(len(self.spots))]
        self.statuses = [True for _ in self.spots]
        self.diffs = [0.0 for _ in self.spots]
        self.diff_by_spot_id = dict(zip(self.spot_ids, self.diffs))

    def process(self, frame):
        if frame is None:
            return None, self.status_dict
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            frame = frame[y1:y2, x1:x2]

        should_detect = self.frame_number % self.step == 0
        if should_detect:
            indexes = self._indexes_to_refresh(frame)
            for spot_index in indexes:
                x1, y1, w, h = self._scaled_box(self.spots[spot_index], self.decision_box_scale)
                spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]
                self.statuses[spot_index] = empty_or_not(spot_crop)
            self.previous_frame = frame.copy()

        self.frame_number += 1
        return frame.copy(), self.status_dict

    @property
    def status_dict(self) -> dict[str, bool]:
        return dict(zip(self.spot_ids, self.statuses))

    def draw(self, frame, sessions=None, billing=None):
        sessions = sessions or {}
        for spot_id, spot, is_empty in zip(self.spot_ids, self.spots, self.statuses):
            x1, y1, w, h = spot
            color = (0, 180, 80) if is_empty else (30, 30, 220)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

            label = spot_id
            session = sessions.get(spot_id)
            if session is not None:
                label = f"{spot_id} {session.vehicle_id}"
                if billing is not None:
                    label = f"{label} {int(billing.fee_for(session))} TL"

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        available = sum(self.statuses)
        total = len(self.statuses)
        cv2.rectangle(frame, (70, 20), (620, 85), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Bos yer: {available} / {total}",
            (95, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        return frame

    def draw_spots(self, frame):
        for spot, is_empty in zip(self.spots, self.statuses):
            x1, y1, w, h = spot
            color = (0, 255, 0) if is_empty else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        return frame

    def _indexes_to_refresh(self, frame):
        if self.previous_frame is None:
            return range(len(self.spots))

        for spot_index, spot in enumerate(self.spots):
            x1, y1, w, h = self._scaled_box(spot, self.decision_box_scale)
            spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]
            previous_crop = self.previous_frame[y1 : y1 + h, x1 : x1 + w, :]
            self.diffs[spot_index] = self._calc_diff(spot_crop, previous_crop)

        max_diff = np.amax(self.diffs)
        self.diff_by_spot_id = dict(zip(self.spot_ids, self.diffs))
        if max_diff == 0:
            return []
        return [i for i in np.argsort(self.diffs) if self.diffs[i] / max_diff > self.change_threshold]

    @staticmethod
    def _calc_diff(im1, im2):
        return np.abs(np.mean(im1) - np.mean(im2))

    @staticmethod
    def _scaled_box(box, scale: float):
        x, y, w, h = box
        scaled_w = max(1, int(w * scale))
        scaled_h = max(1, int(h * scale))
        scaled_x = int(x + (w - scaled_w) / 2)
        scaled_y = int(y + (h - scaled_h) / 2)
        return scaled_x, scaled_y, scaled_w, scaled_h
