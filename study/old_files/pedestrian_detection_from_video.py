# -*- coding: UTF-8 -*-
# http://qiita.com/Algebra_nobu/items/a488fdf8c41277432ff3
import cv2
import os
import numpy

# カスケードファイルと動画ファイルの読み込み
classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('pedestrian_downloaded_from_opencv_samples.mpeg')

while(True):

    # 動画からフレームを取得する
    # frame: 動画から抽出した1フレーム
    ret, frame = cap.read()
    
    # 歩行者を検出する
    # 編集するのはこの処理だけ．ここで自動車を検出できれば良い
    # 入力：frame
    # 出力：矩形のリスト
    pedestrians = classifier.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 2, minSize = (1, 1))
    
    # object detection
    # pedestrians = numpy.array([[0, 0, 0, 0]])
    # オブジェクトを検出したら，そのオブジェクトを囲むような長方形の情報を pedestrians に追加する 
    # 検出したオブジェクトの画像と紐つけておくべき
    # pedestrians = numpy.append(pedestrians, [[134, 123, 94, 54]], axis=0)
    # pedestrians = numpy.append(pedestrians, [[534, 123, 34, 54]], axis=0)

    # feature extraction
    # 検出したオブジェクトの画像を入力したら，特徴量が帰ってくるような関数を作る
    # feature1 = feature1(detected_object)
    # feature2 = feature2(detected_object)
    # feature3 = feature3(detected_object)

    # object classification
    # 検出したオブジェクトを分類して，適切なラベルを判定する
    # label = classifier(feature1, feature2, feature3)

    color = (255, 255, 255)
    text = 'p'
    font = cv2.FONT_HERSHEY_PLAIN

    # 検出した歩行者を矩形で囲む
    for pedestrian in pedestrians:
        cv2.rectangle(frame, tuple(pedestrian[0:2]), tuple(pedestrian[0:2] + pedestrian[2:4]), color, thickness = 2)
        cv2.putText(frame, text, (pedestrian[0], pedestrian[1] - 10), font, 2, color, 2, cv2.LINE_AA)
        
    cv2.imshow("Show FLAME Image", frame)
 
    # qを押下したら終了する
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()