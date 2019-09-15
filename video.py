# -*- coding: UTF-8 -*-

import cv2
import os
import const
import util.image as image

TYPE_DETECTION = 10

class Video:
    def __init__(self, path):
        self.file_name = os.path.basename(path)
        self.video = self.open(path)
        self.current_color = self.__getNextFrame()
        self.current_gray = image.cvt2Gray(self.current_color)
        self.before_gray = self.current_gray.copy().astype('float')

    def open(self, path):
        return cv2.VideoCapture(path)

    def play(self, type=None):
        while(self.current_color is not None):

            if type is TYPE_DETECTION:
                threshold = image.subtract(self.before_gray, self.current_gray)
                contours, heirarchy = image.findContours(threshold)

                for contour in contours:
                    if const.MIN_AREA < cv2.contourArea(contour) < const.MAX_AREA:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(self.current_color, (x, y), (x+w, y+h), const.RECT_COLOR_TRACK, 2)

            cv2.imshow(self.file_name, self.current_color)

            if cv2.waitKey(1) == ord('q'):
                break

            self.moveToNextFrame()

        cv2.destroyAllWindows()

    def playWithDetection(self):
        self.play(type=TYPE_DETECTION)

    def close(self):
        self.video.release()

    def __getNextFrame(self):
        hasNext, frame = self.video.read()
        return frame

    def moveToNextFrame(self):
        self.current_color = self.__getNextFrame()
        self.current_gray = image.cvt2Gray(self.current_color)

if __name__ == "__main__":
    video = Video(const.VIDEO_PATH)
    video.playWithDetection()
    video.close()
