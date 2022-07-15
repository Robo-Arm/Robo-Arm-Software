import cv2 as cv

def capture_frame():
    vid_cap = cv.VideoCapture(0)
    vid_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    vid_cap.set(cv.CAP_PROP_CONVERT_RGB, 0)
    vid_cap.set(cv.CAP_PROP_FPS, 30)

    ret, frame = vid_cap.read()
    cv.imwrite("./c1.png", frame)
    cv.destroyAllWindows()
    vid_cap.release()

if __name__ == "__main__":
    capture_frame()

