# -*- coding: UTF-8 -*-

import cv2
from const import Const
from image import Image

class Video:
    def __init__(self, path):
        self.video = self.open(path)
        self.next_frame = self.__getNextFrame()
        self.current_frame = self.next_frame.copy().astype('float')

    def open(self, path):
        return cv2.VideoCapture(path)

    def close(self):
        self.video.release()

    def __getNextFrame(self):
        hasNext, next_frame = self.video.read()

        if not hasNext:
            return next_frame

        return cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    def moveToNextFrame(self):
        self.next_frame = self.__getNextFrame()

if __name__ == "__main__":
    video = Video(Const.VIDEO_PATH)
    image = Image()
    image.show('current', video.current_frame)
    image.show('next', video.next_frame)
    video.moveToNextFrame()
    image.show('current', video.current_frame)
    image.show('next', video.next_frame)
     
