import cv2 as cv
import numpy as np
import imutils
from typing import List, Tuple
from distance_utils import get_focal_length, get_distance_to_camera

# OpenCV Filtering Constants
LOWER_GREEN_THRESHOLD = np.array([29, 86, 6])
UPPER_GREEN_THRESHOLD = np.array([64, 255, 255])
YELLOW = (0, 255, 255)
MIN_RADIUS_THRESHOLD = 1
WINDOW_NAME = "Video"

# OpenCV Distance Constants
DIST_FONT = cv.FONT_HERSHEY_SIMPLEX
DIST_ORG = (25, 40)
DIST_FONT_SCALE = 1
DIST_THICKNESS = 1

# Focal Length Calibration Constants
KNOWN_RADIUS_CM = 2
KNOWN_DISTANCE_CM = 30

# Initialize the video capture source
def init_video_capture()->cv.VideoCapture:
    # Assume video source is from /dev/video0 for V4L2
    vid_cap = cv.VideoCapture(0)
    vid_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    vid_cap.set(cv.CAP_PROP_CONVERT_RGB, 0)
    vid_cap.set(cv.CAP_PROP_FPS, 30)
    return vid_cap

# Clean up video capture resource
def destroy_video_capture(vid_cap: cv.VideoCapture):
    vid_cap.release()
    cv.destroyAllWindows()

# Blur given frame and convert to colour mask
def convert_frame_to_mask(frame: np.ndarray)->np.ndarray:
    blurred_frame = cv.GaussianBlur(frame, (17, 17), 0)
    hsv_frame = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_frame, LOWER_GREEN_THRESHOLD, UPPER_GREEN_THRESHOLD)
    return mask

# Grab contours given a mask
def get_contours(mask: np.ndarray)->List[np.ndarray]:
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)

# Given contours, find the center and the radius of approximated circle
def get_circle_from_contours(contours: List[np.ndarray])->Tuple[Tuple[int], int]:
    c = max(contours, key=cv.contourArea)
    ((x, y), radius) = cv.minEnclosingCircle(c)
    return (int(round(x)), int(round(y))), int(radius)

# Given a frame, return if found, center, and radius
def get_circle_from_frame(frame: np.ndarray)->Tuple[bool, Tuple[int], int]:
    # Convert input frame to colour mask
    mask = convert_frame_to_mask(frame)
    # Grab contours from mask
    contours = get_contours(mask)

    # Find circle if contours exist
    if not len(contours):
        return (False, (0, 0), 0)
    center, radius = get_circle_from_contours(contours)
    return True, center, radius 

# Determine focal length from reference image
def calibrate_focal_length()->float:
    ref_frame = cv.imread("assets/ref_30cm.png")
    found, center, radius = get_circle_from_frame(ref_frame)
    return get_focal_length(radius, KNOWN_DISTANCE_CM, KNOWN_RADIUS_CM)

# Run main ball detection algorithm
def start_ball_detection():
    focal_length = calibrate_focal_length()
    print(focal_length)
    vid_cap = init_video_capture()
    while True:
        # Capture video frame
        ret, frame = vid_cap.read()
        if not ret:
            break

        found, center, radius = get_circle_from_frame(frame)
        # Find center based on found contours
        if found and radius > MIN_RADIUS_THRESHOLD:
            cv.circle(frame, center, radius, YELLOW, 2)
            # Calculate distance to camera
            distance = get_distance_to_camera(KNOWN_RADIUS_CM, focal_length, radius)
            cv.putText(frame, str(round(distance, 2)) + "cm", DIST_ORG, DIST_FONT, DIST_FONT_SCALE, YELLOW, DIST_THICKNESS, cv.LINE_AA)

        # Display input frame 
        cv.imshow("video", frame)
            
        if cv.waitKey(1) and 0xFF == ord("q"):
            break

    destroy_video_capture(vid_cap)

if __name__ == "__main__":
    start_ball_detection()

