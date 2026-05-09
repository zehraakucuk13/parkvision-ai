from pathlib import Path
import pickle

import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False

BASE_DIR = Path(__file__).resolve().parent
MODEL = pickle.load(open(BASE_DIR / "model.p", "rb"))


def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = cv2.resize(spot_bgr, (15, 15), interpolation=cv2.INTER_AREA)
    img_resized = img_resized.astype("float32") / 255.0
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if int(y_output[0]) == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots
