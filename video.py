# -*- coding: UTF-8 -*-

import cv2
import const
from image import Image

class Video:
    def __init__(self, path):
        self.video = self.open(path)
        self.current_color = self.__getNextFrame()
        self.current_gray = self.__cvt2Gray(self.current_color)
        self.before_gray = self.current_gray.copy().astype('float')

    def open(self, path):
        return cv2.VideoCapture(path)

    def play(self):
        while(self.current_color is not None):
            cv2.imshow('result', video.current_color)

            if cv2.waitKey(1) == ord('q'):
                break

            self.moveToNextFrame()

        cv2.destroyAllWindows()

    def close(self):
        self.video.release()

    def __getNextFrame(self):
        hasNext, frame = self.video.read()
        return frame

    def __cvt2Gray(self, frame):
        if frame is None:
            return frame

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def moveToNextFrame(self):
        self.current_color = self.__getNextFrame()
        self.current_gray = self.__cvt2Gray(self.current_color)

if __name__ == "__main__":
    video = Video(const.VIDEO_PATH)
    video.play()
    video.close()
