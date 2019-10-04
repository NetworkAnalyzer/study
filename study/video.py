# -*- coding: UTF-8 -*-

import cv2
import os
import const
from object import Object
import util.image as image

TYPE_PALY = 0
TYPE_DETECTION = 10
TYPE_DISPLAY = 20
TYPE_SELECTION = 30
TYPE_ALL_SAVING = 40


class Video:
    def __init__(self, path):
        self.file_name = os.path.basename(path)
        self.video = self.open(path)
        self.current_color = self.__getNextFrame()
        self.current_gray = image.cvt2Gray(self.current_color)
        self.before_gray = self.current_gray.copy().astype('float')

    def open(self, path):
        return cv2.VideoCapture(path)

    def play(self, type=TYPE_PALY, start_from=0):
        for i in range(start_from):
            self.moveToNextFrame()

        classifier = cv2.CascadeClassifier(const.CASCADE_PATH)

        cnt = 1
        while self.current_color is not None:

            if type >= TYPE_DETECTION:

                objects = classifier.detectMultiScale(
                    self.current_color,
                    scaleFactor=1.05,
                    minNeighbors=2,
                    minSize=(10, 10),
                )

                for object in objects:
                    (x, y) = tuple(object[0:2])
                    (w, h) = tuple(object[2:4])

                    if type >= TYPE_DISPLAY:
                        object = Object(x, y, w, h)
                        object.image = video.current_gray[y : y + h, x : x + w]
                        image.show('trimmed', object.image, gray=True)

                    if type >= TYPE_SELECTION:
                        k = cv2.waitKey(0) & 0xFF
                        if k == ord('t') or k == ord('c'):
                            cv2.imwrite(
                                'image/{0}_{1}.png'.format(cnt, chr(k)), object.image
                            )
                            cnt += 1

                    cv2.rectangle(
                        self.current_color,
                        (x, y),
                        (x + w, y + h),
                        const.RECT_COLOR_TRUCK,
                        2,
                    )

            cv2.imshow(self.file_name, self.current_color)

            if cv2.waitKey(1) == ord('q'):
                break

            self.moveToNextFrame()

        cv2.destroyAllWindows()

    def playWithDetection(self, start_from=0):
        self.play(type=TYPE_DETECTION, start_from=start_from)

    def playWithDisplay(self, start_from=0):
        self.play(type=TYPE_DISPLAY, start_from=start_from)

    def playWithSelection(self, start_from=0):
        self.play(type=TYPE_SELECTION, start_from=start_from)

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
    video.playWithSelection(start_from=1000)
    video.close()
