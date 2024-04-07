###########################
#                         #
#  Файл с функцией main   #
#                         #
###########################


import numpy as np
import cv2 as cv
from util.picture_edit import VideoEditor

cap = cv.VideoCapture("Traffic IP Camera video (1).mp4")

bg_subs = cv.createBackgroundSubtractorMOG2()
history = 300
learning_rate = 1.0 / history

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = VideoEditor.frame_edit(frame, 1)
    mask = bg_subs.apply(frame, learningRate=learning_rate)
    out = frame.copy()
    # result = cv.bitwise_and(out, out, mask=mask)
    # cv.imshow("res", mask)

    ret, thresh1 = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)
    filter_im = VideoEditor.filter_im(thresh1, 5)
    edited = VideoEditor.mask_edit(filter_im, 3, 3)
    # cv.imshow("1", mask)
    # cv.imshow("2", filter_im)
    # cv.imshow("3", edited)

    VideoEditor().find_ctr(edited, out)
    cv.imshow("input", out)
    c = cv.waitKey(10)
    if c == 27:
        break
cap.release()
cv.destroyAllWindows()