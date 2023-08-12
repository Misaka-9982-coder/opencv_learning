#!/usr/bin/env python3
# encoding:utf-8


import sys
import cv2 as cv

def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Error opening the video source.')
        sys.exit()

    # Get the width and height of the video frame
    frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    fps = 25

    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    writer = cv.VideoWriter("01-图像的基本操作/myvideo.avi", fourcc, fps, frame_size)

    if not writer.isOpened():
        print("Error creating video writer.")
        sys.exit()

    frame_count = 0
    while True:
        ret, im = cap.read()
        if not ret:
            print('No image read.')
            break

        writer.write(im)  # write the frame to the file

        # Show the video frame
        cv.imshow('Live', im)

        # Wait for 30 ms, if there is a key press then break the loop
        if cv.waitKey(30) >= 0:
            break

        frame_count += 1
        if frame_count > (3 * fps):  # if we have recorded for 5 seconds, break the loop
            break

    # Release the writer and capture object
    writer.release()
    cap.release()

    cv.destroyAllWindows()


if __name__  == '__main__':
    main()
