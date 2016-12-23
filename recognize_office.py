#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#載入必要模組
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import numpy as np

#使用參數方式傳入Training和Test的dataset
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())

#data用來存放HOG資訊，labels則是存放對應的標籤
print "[INFO] extracting features..."
data = []
labels = []

#產生mask圖層
mask = np.zeros((500,500), dtype="uint8")
#定義四個角落的位置
cv2.rectangle(mask, (1, 63), (174, 262), 255, -1)
cv2.rectangle(mask, (1, 292), (180, 500), 255, -1)
cv2.rectangle(mask, (300, 60), (480, 265), 255, -1)
cv2.rectangle(mask, (300, 295), (480, 500), 255, -1)

#依序讀取training dataset中的圖檔
for imagePath in paths.list_images(args["training"]):
     #將資料夾的名稱取出作為該圖檔的標籤
    make = imagePath.split("/")[-2]

     #載入圖檔
    image = cv2.imread(imagePath)
     #將Mask疊加到圖檔
    masked = cv2.bitwise_and(image, image, mask=mask)
     #轉為灰階
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
     #取得其HOG資訊及視覺化圖檔
    H = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
            cells_per_block=(2, 2), transform_sqrt=True, visualise=False)

     #將HOG資訊及標籤分別放入data及labels陣列
    data.append(H)
    labels.append(make)

#訓練資料中的10%( test_size=0.1)用於驗証使用，以便計算出最佳的Ｋ值
(data, valData, labels, valLabels) = train_test_split(data, labels, test_size=0.1, random_state=84)

#顯示訓練及驗証用資料的筆數
print("training data points: {}".format(len(labels)))
print("validation data points: {}".format(len(valLabels)))

kVals = range(1, 30, 2)
#存放各K值的正確率
accuracies = []

#計算從1~30之間的K值（K必須為奇數）
for k in xrange(1, 30, 2):
    #宣告KNN模型，n_ neighbors值為K 
    model = KNeighborsClassifier(n_neighbors=k)
    #使用訓練資料集data及labels來訓練
    model.fit(data, labels)
    #使用驗証資料集valData及valLabels來驗証，取得並顯示其正確率
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    #將此正確率放到score陣列中，因為最後要取出最高正確率的那個
    accuracies.append(score)

#找到有最高正確率的Ｋ值並顯示出來
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

maxK = kVals[i]
print "[INFO] evaluating..."

#這兩個Array用來放置我們的Test dataset
testLabels = []
testData = []

#依序讀取Test Dataset圖檔
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
    #將資料夾的名稱取出作為該圖檔的標籤，並append到testLabels
    make = imagePath.split("/")[-2]
    testLabels.append(make)
    #載入圖檔
    image = cv2.imread(imagePath)
    #套用Mask
    masked = cv2.bitwise_and(image, image, mask=mask)
    #轉為灰階
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    #取得其HOG資訊及視覺化圖檔，並append到testData
    (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=False, visualise=True)
    testData.append(H)

#宣告KNN模型，n_ neighbors值為剛剛得出的最好的K值
model = KNeighborsClassifier(n_neighbors=maxK)
#用此K值來訓練
model.fit(data, labels)
#用test dataset來預測
predictions = model.predict(testData)
#統計預測結果並印出
print(classification_report(testLabels, predictions))

