import cv2 as cv
import numpy as np
import imutils
from typing import List, Tuple

# Constants
LOWER_GREEN_THRESHOLD = np.array([29, 86, 6])
UPPER_GREEN_THRESHOLD = np.array([64, 255, 255])
GREEN = (0, 255, 255)
MIN_RADIUS_THRESHOLD = 1

# Initialize the video capture source
def init_video_capture()->cv.VideoCapture:
    # Assume video source is from /dev/video0
    vid_cap = cv.VideoCapture(0)
    vid_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    vid_cap.set(cv.CAP_PROP_CONVERT_RGB, 0)
    vid_cap.set(cv.CAP_PROP_FPS, 30)
    return vid_cap

# Clean up video capture resource
def destroy_video_capture(vid_cap: cv.VideoCapture):
    vid_cap.release()
    cv.destroyAllWindows()

# Blur given frame and convert to hue, saturation, value
def convert_frame_to_hsv(frame: np.ndarray)->np.ndarray:
    blurred_frame = cv.GaussianBlur(frame, (17, 17), 0)
    hsv_frame = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)
    return hsv_frame

# Grab contours given a mask
def get_contours(mask: np.ndarray)->List[np.ndarray]:
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)

# Given contours, find the center and the radius of approximated circle
def get_circle_from_contours(contours: List[np.ndarray])->Tuple[Tuple[int], int]:
    c = max(contours, key=cv.contourArea)
    ((x, y), radius) = cv.minEnclosingCircle(c)
    return (int(round(x)), int(round(y))), int(radius)

# Start ball detection algorithm
def start_ball_detection():
    vid_cap = init_video_capture()
    while True:
        # Capture video frame
        ret, frame = vid_cap.read()
        if not ret:
            break

        # Convert input frame to color mask
        hsv_frame = convert_frame_to_hsv(frame)
        mask = cv.inRange(hsv_frame, LOWER_GREEN_THRESHOLD, UPPER_GREEN_THRESHOLD)
        # Grab contours from mask
        contours = get_contours(mask)

        # Find center based on found contours
        if len(contours) > 0:
            center, radius = get_circle_from_contours(contours)
            if radius > MIN_RADIUS_THRESHOLD:
                cv.circle(frame, center, radius, GREEN, 2)

        # Display input frame and mask
        cv.imshow("video", frame)
        cv.imshow("mask", mask)
            
        if cv.waitKey(1) and 0xFF == ord("q"):
            break

    destroy_video_capture(vid_cap)

if __name__ == "__main__":
    start_ball_detection()

