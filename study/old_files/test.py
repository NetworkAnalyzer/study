# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# setting for changing playback speed of a video
WAIT_KEY = 50
# interval of generating subtraction
SECOND = 2
# threshold for binarization
THRESH = 100
# maximum value of pixels
MAXVAL = 255

def gaussian_filter(img):
    
    kernel = np.array([[1/16, 1/8, 1/16],
                       [1/8,  1/4, 1/8 ],
                       [1/16, 1/8, 1/16]])

    return cv2.filter2D(img, -1, kernel)

def generate_subtraction(bg, fg):

    absdiff = cv2.absdiff(bg, fg)

    return cv2.threshold(absdiff, 50, 255, cv2.THRESH_BINARY)[1]

def to_binary(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.threshold(gray, THRESH, MAXVAL, cv2.THRESH_BINARY)[1]

if __name__ == '__main__':

    bg = cv2.imread("21.jpg")
    bg_filterd = gaussian_filter(bg)
    fg = cv2.imread("26.jpg")
    fg_filterd = gaussian_filter(fg)

    subtraction = generate_subtraction(bg_filterd, fg_filterd)
    cv2.imwrite('subtraction.jpg', subtraction)
    binary = to_binary(subtraction)
    cv2.imwrite('binary.jpg', binary)
    
    exit()

    video = cv2.VideoCapture("walking_short.mp4")
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    interval = fps * SECOND

    fg_pos = 1
    bg_pos = fg_pos + interval

    for pos in range(fg_pos, frame_count, 5):
        video.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = video.read()

        if not ret:
            break
        
        cv2.imwrite(str(pos) +".jpg", frame)

        if cv2.waitKey(WAIT_KEY) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()