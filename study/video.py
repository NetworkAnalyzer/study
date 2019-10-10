# -*- coding: UTF-8 -*-

import cv2
import os
from study import const
from study.object import Object
import study.util.image as image


class Video:
    def __init__(self, path):
        self.file_name = os.path.basename(path)
        self.video = self.open(path)
        self.current_color = self.__getNextFrame()
        self.current_gray = image.cvt2Gray(self.current_color)
        self.before_gray = self.current_gray.copy().astype('float')

    def open(self, path):
        return cv2.VideoCapture(path)

    def play(self, save=False, start_from=0):
        for i in range(start_from):
            self.moveToNextFrame()

        classifier = cv2.CascadeClassifier(const.CASCADE_PATH)

        if save is True:
            image_path = os.path.join('image', self.file_name)
            if not os.path.exists(image_path):
                os.makedirs(image_path, exist_ok=True)

        cnt = 1
        while self.current_color is not None:

            objects = classifier.detectMultiScale(
                self.current_color,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(10, 10),
            )

            for object in objects:
                (x, y) = tuple(object[0:2])
                (w, h) = tuple(object[2:4])

                if save is True:
                    cv2.imwrite(os.path.join(image_path, '{0}.png'.format(cnt)), self.current_gray[y : y + h, x : x + w])
                    cnt += 1

                cv2.rectangle(
                    self.current_color,
                    (x, y),
                    (x + w, y + h),
                    const.RECT_COLOR_TRUCK,
                    2,
                )

            if cv2.waitKey(1) == ord('q'):
                break

            self.showFrame()
            self.moveToNextFrame()

        cv2.destroyAllWindows()

    def close(self):
        self.video.release()

    def __getNextFrame(self):
        hasNext, frame = self.video.read()
        return frame

    def moveToNextFrame(self):
        self.current_color = self.__getNextFrame()
        self.current_gray = image.cvt2Gray(self.current_color)

    def showFrame(self):
        cv2.imshow(self.file_name, self.current_color)

def main():
    video = Video(const.VIDEO_PATH)
    video.play(save=True)
    video.close()
