# -*- coding: UTF-8 -*-

import cv2
from video import Video
from image import Image
from const import Const

if __name__ == "__main__":
    def subtract(current_frame, next_frame):
        cv2.accumulateWeighted(next_frame, current_frame, 0.5)
        mdframe = cv2.absdiff(next_frame, cv2.convertScaleAbs(current_frame))
        return cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]

    const = Const()
    video = Video(const.VIDEO_PATH)

    while(video.next_frame is not None):
        thresh = subtract(video.current_frame, video.next_frame)

        cv2.imshow('result', video.next_frame)

        cv2.waitKey(const.DELAY)
        if cv2.waitKey(1) == ord('q'):
            break

        video.moveToNextFrame()

    video.close()
