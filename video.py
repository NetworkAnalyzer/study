# -*- coding: UTF-8 -*-

import cv2
from const import Const
import numpy as np
import matplotlib.pyplot as plt

class Video:
    def __init__(self):
        self.video = self.open(Const.VIDEO_PATH)
        self.current_frame = self.__getNextFrame()
        self.next_frame = self.current_frame.copy().astype('float')

    def open(self, path):
        return cv2.VideoCapture(path)

    def close(self):
        self.video.release()

    def __cvt2Gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def __getNextFrame(self):
        ret, frame = self.video.read()
        return self.__cvt2Gray(frame)

    def toNext(self):
        self.current_frame = self.next_frame
        self.next_frame = self.__getNextFrame()

    def showFrame(self, title):
        if title is 'current':
            frame = self.current_frame
        elif title is 'next':
            frame = self.next_frame
        else:
            return

        plt.title(title)
        plt.imshow(frame, cmap='gray')
        plt.show()

if __name__ == "__main__":
    video = Video()
    video.showFrame('current')
    video.showFrame('next')
    video.toNext()
    video.showFrame('current')
    video.showFrame('next')
     
