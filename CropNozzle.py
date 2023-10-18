import os
import string
import cv2 as cv
import numpy as np
import random as rn

def recursive_string(folder):
    s = ''.join(rn.choices(list(string.ascii_lowercase), k=10))
    if os.path.exists(os.path.join(folder, s + '.png')):
        s = recursive_string(folder)
    return s

DESTINATION = 'Crops_NEW'
THRESHOLD = 0.85
VIDEO = r'Nozzle\no3.mp4'
LABEL = 'NO'
NEEDLE = 'template_no.png'
print(VIDEO, LABEL)

os.makedirs(DESTINATION, exist_ok=True)
os.makedirs(os.path.join(DESTINATION, 'YES'), exist_ok=True)
os.makedirs(os.path.join(DESTINATION, 'NO'), exist_ok=True)

cap = cv.VideoCapture(VIDEO)
needle = cv.imread(NEEDLE)
needle = cv.cvtColor(needle, cv.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    haystack = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    for aspect_ratio in np.arange(0.4, 2.0, 0.2):
        needle_resized = cv.resize(needle, None, None, aspect_ratio, aspect_ratio, cv.INTER_CUBIC)
        needle_h, needle_w = needle_resized.shape
        result = cv.matchTemplate(haystack, needle_resized, cv.TM_CCOEFF_NORMED)
        ymins, xmins = np.where(result >= THRESHOLD)
        ymaxs = ymins + needle_h
        xmaxs = xmins + needle_w
        bboxes = np.stack([xmins, ymins, xmaxs, ymaxs], axis=-1)
        if bboxes.shape[0] > 0:
            max_area = 0
            index = 0
            for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
                area = (xmax - xmin) * (ymax - ymin)
                if area > max_area:
                    index = i
                    max_area = area
            xmin, ymin, xmax, ymax = bboxes[index]
            bbox = frame[ymin:ymax, xmin:xmax, :]
            cv.imwrite(os.path.join(DESTINATION, LABEL, recursive_string(os.path.join(DESTINATION, LABEL)) + '.png'), bbox)
cap.release()
