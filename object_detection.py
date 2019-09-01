# -*- coding: UTF-8 -*-

import cv2

from video import Video
from image import Image
from object import Object
import const

if __name__ == "__main__":
    def subtract(before_gray, current_gray):
        cv2.accumulateWeighted(current_gray, before_gray, 0.5)
        mdframe = cv2.absdiff(current_gray, cv2.convertScaleAbs(before_gray))
        return cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]

    def findContours(image):
        return cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    video = Video(const.VIDEO_PATH)
    image = Image()

    while(video.current_color is not None):
        threshold = subtract(video.before_gray, video.current_gray)
        contours, heirarchy = findContours(threshold)

        for contour in contours:
            if cv2.contourArea(contour) > const.MIN_AREA:
                x, y, w, h = cv2.boundingRect(contour)

                object = Object(x, y, w, h)
                print(object.compactness)
                print(object.hwr)

                top_left = (x, y)
                bottom_right = (x + w, y + h)

                object.image = video.current_gray[x:x+w, y:y+h]
                print(object.image)
                cv2.imwrite('')
                image.show('trimming', object.image, gray=True)

                cv2.rectangle(video.current_color, top_left, bottom_right, const.RECT_COLOR, 2)

        cv2.imshow('result', video.current_color)

        cv2.waitKey(const.DELAY)
        if cv2.waitKey(1) == ord('q'):
            break

        video.moveToNextFrame()

    video.close()
    cv2.destroyAllWindows()