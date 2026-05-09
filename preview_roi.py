from pathlib import Path

import cv2

from detector import ParkingDetector


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "samples" / "parking_1920_1080.mp4"
MASK_PATH = BASE_DIR / "mask_1920_1080.png"
ROI = (1180, 120, 1380, 940)


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame from {VIDEO_PATH}")

    detector = ParkingDetector(MASK_PATH, step=1)
    preview = detector.draw_spots(frame.copy())
    roi_preview = preview.copy()
    x1, y1, x2, y2 = ROI
    cv2.rectangle(roi_preview, (x1, y1), (x2, y2), (255, 0, 0), 5)
    cv2.putText(
        roi_preview,
        f"ROI {ROI}",
        (x1, max(40, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3,
    )
    roi_crop = preview[y1:y2, x1:x2]

    cv2.imwrite(str(BASE_DIR / "roi_reference_frame.jpg"), frame)
    cv2.imwrite(str(BASE_DIR / "roi_slots_preview.jpg"), preview)
    cv2.imwrite(str(BASE_DIR / "roi_marked_preview.jpg"), roi_preview)
    cv2.imwrite(str(BASE_DIR / "roi_crop_preview.jpg"), roi_crop)

    print(BASE_DIR / "roi_reference_frame.jpg")
    print(BASE_DIR / "roi_slots_preview.jpg")
    print(BASE_DIR / "roi_marked_preview.jpg")
    print(BASE_DIR / "roi_crop_preview.jpg")


if __name__ == "__main__":
    main()
