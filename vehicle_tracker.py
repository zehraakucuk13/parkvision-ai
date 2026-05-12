from dataclasses import dataclass

import cv2


@dataclass
class ExitingVehicleTrack:
    vehicle_id: str
    bbox: tuple[int, int, int, int]
    tracker: object
    missed_frames: int = 0


class ParkingVehicleTracker:
    def __init__(
        self,
        max_missed_frames: int = 45,
        tracker_box_scale: float = 0.7,
        exit_motion_threshold: float = 3.0,
    ):
        self.max_missed_frames = max_missed_frames
        self.tracker_box_scale = tracker_box_scale
        self.exit_motion_threshold = exit_motion_threshold
        self._vehicle_counter = 0
        self._previous_statuses: dict[str, bool] = {}
        self._previous_frame = None
        self.parked_vehicle_ids: dict[str, str] = {}
        self.entry_vehicle_ids: set[str] = set()
        self.exiting_tracks: list[ExitingVehicleTrack] = []
        self.completed_vehicle_ids: list[str] = []
        self._initialized = False

    def update(
        self,
        statuses: dict[str, bool],
        spot_boxes: dict[str, list[int]],
        frame,
        motion_by_spot_id: dict[str, float] | None = None,
    ) -> None:
        motion_by_spot_id = motion_by_spot_id or {}
        self._update_exiting_tracks(frame)

        for spot_id, is_empty in statuses.items():
            was_empty = self._previous_statuses.get(spot_id, True)

            if not is_empty and spot_id not in self.parked_vehicle_ids:
                vehicle_id = self._next_vehicle_id()
                self.parked_vehicle_ids[spot_id] = vehicle_id
                if self._initialized:
                    self.entry_vehicle_ids.add(vehicle_id)

            if is_empty and not was_empty and spot_id in self.parked_vehicle_ids:
                motion = motion_by_spot_id.get(spot_id, 0.0)
                if self._previous_frame is not None and motion >= self.exit_motion_threshold:
                    vehicle_id = self.parked_vehicle_ids.pop(spot_id)
                    self._start_exiting_track(vehicle_id, spot_boxes[spot_id], self._previous_frame)

        self._previous_statuses = statuses.copy()
        self._previous_frame = frame.copy()
        self._initialized = True

    def draw(self, frame, spot_boxes: dict[str, list[int]]):
        for spot_id, vehicle_id in self.parked_vehicle_ids.items():
            x, y, w, h = spot_boxes[spot_id]
            self._draw_label(frame, vehicle_id, x + 4, y + 14, (255, 255, 255))

        for track in self.exiting_tracks:
            x, y, w, h = track.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            self._draw_label(frame, track.vehicle_id, x + 4, y + 14, (255, 0, 0))

        return frame

    def rows(self) -> list[dict]:
        parked = [
            {
                "Araç ID": vehicle_id,
                "Durum": "Giriş yaptı" if vehicle_id in self.entry_vehicle_ids else "Park halinde",
            }
            for vehicle_id in sorted(self.parked_vehicle_ids.values())
        ]
        exiting = [
            {"Araç ID": track.vehicle_id, "Durum": "Çıkıyor"}
            for track in self.exiting_tracks
        ]
        completed = [
            {"Araç ID": vehicle_id, "Durum": "Çıkış yaptı"}
            for vehicle_id in self.completed_vehicle_ids[-20:]
        ]
        return parked + exiting + completed

    def complete_exiting_tracks(self) -> None:
        for track in self.exiting_tracks:
            if track.vehicle_id not in self.completed_vehicle_ids:
                self.completed_vehicle_ids.append(track.vehicle_id)
        self.exiting_tracks = []

    def _next_vehicle_id(self) -> str:
        self._vehicle_counter += 1
        return f"V-{self._vehicle_counter:04d}"

    def _start_exiting_track(self, vehicle_id: str, box: list[int], frame) -> None:
        x, y, w, h = self._scaled_box(box, self.tracker_box_scale)
        tracker = cv2.TrackerMIL_create()
        tracker.init(frame, (x, y, w, h))
        self.exiting_tracks.append(
            ExitingVehicleTrack(vehicle_id=vehicle_id, bbox=(x, y, w, h), tracker=tracker)
        )

    def _update_exiting_tracks(self, frame) -> None:
        active_tracks = []
        frame_h, frame_w = frame.shape[:2]

        for track in self.exiting_tracks:
            ok, bbox = track.tracker.update(frame)
            if ok:
                x, y, w, h = [int(value) for value in bbox]
                old_x, old_y, old_w, old_h = track.bbox
                old_center = (old_x + old_w / 2, old_y + old_h / 2)
                new_center = (x + w / 2, y + h / 2)
                center_jump = ((new_center[0] - old_center[0]) ** 2 + (new_center[1] - old_center[1]) ** 2) ** 0.5
                size_ratio = (w * h) / max(1, old_w * old_h)

                if center_jump > max(old_w, old_h) * 1.8 or size_ratio < 0.35 or size_ratio > 2.8:
                    track.missed_frames += 1
                else:
                    track.bbox = (x, y, w, h)
                    track.missed_frames = 0
            else:
                track.missed_frames += 1

            x, y, w, h = track.bbox
            is_outside = x + w < 0 or y + h < 0 or x > frame_w or y > frame_h
            if track.missed_frames <= self.max_missed_frames and not is_outside:
                active_tracks.append(track)
            elif track.vehicle_id not in self.completed_vehicle_ids:
                self.completed_vehicle_ids.append(track.vehicle_id)

        self.exiting_tracks = active_tracks

    @staticmethod
    def _draw_label(frame, text: str, x: int, y: int, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.36
        thickness = 1
        x = max(0, x)
        y = max(12, y)

        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)

    @staticmethod
    def _scaled_box(box, scale: float):
        x, y, w, h = box
        scaled_w = max(1, int(w * scale))
        scaled_h = max(1, int(h * scale))
        scaled_x = int(x + (w - scaled_w) / 2)
        scaled_y = int(y + (h - scaled_h) / 2)
        return scaled_x, scaled_y, scaled_w, scaled_h
