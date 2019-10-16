# -*- coding: UTF-8 -*-

import cv2
import os
import numpy as np
from study.util.path import image_path
from study import const
from study.object import Object
import study.util.image as image
import study.twmeggs.anfis.anfis as twmeggs

class Video:
    def __init__(self, path):
        self.path = path
        self.file_name = os.path.splitext(os.path.basename(self.path))[0]
        self.video = self.open()
        self.current_color = self.__getNextFrame()
        self.current_gray = image.cvt2Gray(self.current_color)
        self.before_gray = self.current_gray.copy().astype('float')

    def open(self):
        return cv2.VideoCapture(self.path)

    def play(self, save, classify, anfises, start_from=0):
        if classify:
            if anfises is None:
                print('Error: you need to set anfis object')
                exit()
            elif anfises['car'].isTrained is False and anfises['truck'].isTrained is False:
                print('Error: you need to train anfis before')
                exit()

        for _ in range(start_from):
            self.moveToNextFrame()

        classifier = cv2.CascadeClassifier(const.get('CASCADE_PATH'))

            os.makedirs(image_path(self.file_name), exist_ok=True)
        if save:

        cnt = 1
        rectangle_color = None
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

                if classify:
                    features = Object(self.current_gray[y : y + h, x : x + w])
                    feature = features.get(const.get('FEATURE'))

                    result = [
                        twmeggs.predict(anfises['car'], np.array([[feature]]))[0][0] > 0.5,
                        twmeggs.predict(anfises['truck'], np.array([[feature]]))[0][0] > 0.5,
                    ]
                    
                    if result == [True, False]:
                        rectangle_color = const.get('RECT_COLOR_CAR')
                    elif result == [False, True]:
                        rectangle_color = const.get('RECT_COLOR_TRUCK')

                    cv2.imwrite(image_path(self.file_name + '/{0}.png'.format(cnt)), self.current_gray[y : y + h, x : x + w])
                if save:
                    cnt += 1

                if rectangle_color is not None:
                    cv2.rectangle(
                        self.current_color,
                        (x, y),
                        (x + w, y + h),
                        rectangle_color,
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

def main(path=const.get('VIDEO_PATH'), save=False, classify=False, anfises=None):
    video = Video(path)
    video.play(save=bool(save), classify=bool(classify), anfises=anfises)
    video.close()
