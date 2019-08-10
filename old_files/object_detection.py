# -*- coding: UTF-8 -*-
# https://ensekitt.hatenablog.com/entry/2018/06/11/200000
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from feature_extraction import extract_features

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

DELAY = 40 # ms
MIN_AREA = 500 # height * width (px)
COLOR = (0, 255, 0) # rectangle color

def classify(features):
    return True

if __name__ == '__main__':
    cap = cv2.VideoCapture(os.path.join(BASE_DIR, os.getenv('VIDEO_PATH')))
    before = None
    obj_now = None

    while(True):
        # 動画から1フレームを取得する
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1つ前のフレームを取得する
        if before is None:
            before = gray.copy().astype('float')
            continue

        # 差分画像を生成する
        cv2.accumulateWeighted(gray, before, 0.5)
        mdframe = cv2.absdiff(gray, cv2.convertScaleAbs(before))
        thresh = cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]

        # オブジェクトを矩形で囲む
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sum_feature1 = sum_feature2 = count = 0
        
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 今の仕様だと，同じcontours内の前後のobj情報を比較していることになる
                # 本来は，次のframeと比較しないといけないのに
                c_x = x + w / 2
                c_y = y + h / 2
                obj_next = np.array([x, y, w, h, c_x, c_y])

                if obj_now is None:
                    obj_now = obj_next
                features = extract_features(obj_now, obj_next)
                sum_feature1 += features[0]
                sum_feature2 += features[1]
                count += 1

                print(features)
                if classify(features):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR, 2)
            
                obj_now = obj_next

        cv2.imshow('result', frame)

        cv2.waitKey(DELAY) # 動画の再生速度を調整するための待ち時間

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    print(sum_feature1 / count)
    print(sum_feature2 / count)

    cap.release()
    cv2.destroyAllWindows()