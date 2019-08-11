# -*- coding: UTF-8 -*-

import cv2
from const import Const
from image import Image

class Video:
    def __init__(self):
        self.video = self.open(Const.VIDEO_PATH)
        self.imageInstance = Image()
        self.current_frame = self.__getNextFrame()
        self.next_frame = self.current_frame.copy().astype('float')

    def open(self, path):
        return cv2.VideoCapture(path)

    def close(self):
        self.video.release()

    def __getNextFrame(self):
        ret, frame = self.video.read()
        return self.imageInstance.cvt2Gray(frame)

    def toNext(self):
        self.current_frame = self.next_frame
        self.next_frame = self.__getNextFrame()

if __name__ == "__main__":
    video = Video()
    video.imageInstance.show('current', video.current_frame)
    video.imageInstance.show('next', video.next_frame)
    video.toNext()
    video.imageInstance.show('current', video.current_frame)
    video.imageInstance.show('next', video.next_frame)
     
