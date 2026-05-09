import cv2

from detector import ParkingDetector
from vehicle_tracker import ParkingVehicleTracker


MASK_PATH = "./mask_1920_1080.png"
VIDEO_PATH = "./samples/parking_1920_1080.mp4"
ROI = (260, 419, 1200, 1040)


def main():
    detector = ParkingDetector(MASK_PATH, step=30, roi=ROI)
    vehicle_tracker = ParkingVehicleTracker()
    spot_boxes = dict(zip(detector.spot_ids, detector.spots))
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, statuses = detector.process(frame)
        vehicle_tracker.update(statuses, spot_boxes, annotated, detector.diff_by_spot_id)
        annotated = detector.draw_spots(annotated)
        annotated = vehicle_tracker.draw(annotated, spot_boxes)

        cv2.namedWindow("ParkVision AI", cv2.WINDOW_NORMAL)
        cv2.imshow("ParkVision AI", annotated)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
