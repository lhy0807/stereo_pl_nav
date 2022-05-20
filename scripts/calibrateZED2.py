import cv2

if __name__ == "__main__":
    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    FRAME_WIDTH = int(2560/2)

    _, frame = vid.read()