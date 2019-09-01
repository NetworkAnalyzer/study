# -*- coding: UTF-8 -*-

import cv2

from video import Video
from image import Image
import const

if __name__ == "__main__":
    video = Video(const.VIDEO_PATH)
    image = Image()
    cnt = 1

    while(video.current_color is not None):
        threshold = image.subtract(video.before_gray, video.current_gray)
        contours, heirarchy = image.findContours(threshold)

        for contour in contours:
            if const.MIN_AREA < cv2.contourArea(contour) < const.MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)

                current_image = video.current_color
                cv2.rectangle(current_image, (x, y), (x + w, y + h), const.RECT_COLOR, 2)
                cv2.imshow('result', current_image)

                trimmed = video.current_gray[y:y+h, x:x+w]
                if trimmed.shape[0] > 0 and trimmed.shape[1] > 0:
                    cv2.imshow('trimmed', trimmed)

                k = cv2.waitKey(0) & 0xFF
                if k == ord('t'):
                    cv2.imwrite('image/trimmed_{0}_track.png'.format(cnt), trimmed)
                    cnt+=1
                elif k == ord('c'):
                    cv2.imwrite('image/trimmed_{0}_car.png'.format(cnt), trimmed)
                    cnt+=1
                    
        video.moveToNextFrame()

        k = cv2.waitKey(1000) & 0xFF
        if k == ord('q'):
            break

    video.close()
    cv2.destroyAllWindows()