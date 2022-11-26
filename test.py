################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

from posixpath import split
from shutil import SpecialFileError
from turtle import circle, numinput
import numpy as np
import cv2
import os
import math

#Relative path to classifier cascade
ClassifierPath = "NoEntryCascade/cascade.xml"
MinLineProbability = 0.7
MaxLineProbability = 6
MinCircleProbability = 4
MinSegmentProbability = 0.3

def loadImages(directory='No_entry', downscale = 1):
    dirs = os.listdir(directory)
    images = {}
    for dir in dirs:
        image = cv2.imread(directory + "/" + dir, 1)
        images[dir] = cv2.resize(image, (int(image.shape[1] / downscale), int(image.shape[0] / downscale)))
    return images

def cleanupImages(images):
    cleaned = {}
    for img_name, image in images.items():
        cleaned[img_name] = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

    return cleaned

def intersectionOverUnion(A, B):
    areaA = A[2] * A[3]
    areaB = B[2] * B[3]
    xStart = max(A[0], B[0])
    xEnd = min(A[0] + A[2], B[0] + B[2])
    yStart = max(A[1], B[1])
    yEnd = min(A[1] + A[3], A[1] + A[3])
    areaIntersection = max(0,(xEnd - xStart) * (yEnd - yStart))
    unionArea = areaA + areaB - areaIntersection
    return float(areaIntersection) / unionArea

def calculateF1(TP, FP, FN):
    if (TP == 0):
        return 0
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    return 2 * precision * recall / (precision + recall)


def useKernel(img, x, y, krnl, size):
    sum = 0
    for j in range(-size + 1, size):
        for i in range(-size + 1, size):
            yPos = (y + j) % img.shape[0]
            xPos = (x + i) % img.shape[1]
            mod = krnl[-j + size - 1][-i + size - 1]
            sum += img[yPos][xPos] * mod

    return sum


def convolute(img, krnl):
    newImage = np.zeros(img.shape)
    krnlSize = math.ceil(krnl.shape[0] / 2)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            newImage[y][x] = useKernel(img, x, y, krnl, krnlSize)

    return newImage

def houghLine(thresholdMag, dir, pResolution, thetaResolution):
    maxP = math.sqrt(thresholdMag.shape[0] ** 2 + thresholdMag.shape[1] ** 2)
    h = np.zeros((pResolution, thetaResolution))
    for y in range(thresholdMag.shape[0]):
        for x in range(thresholdMag.shape[1]):
            d = dir[y][x]
            if thresholdMag[y][x] > 0:
                thetaIndex = int(thetaResolution * (d + math.pi / 2) / math.pi)
                p = x * math.cos(d) + y * math.sin(d)
                if thetaIndex >= 0 and thetaIndex < thetaResolution:
                    h[int(pResolution * p / maxP)][thetaIndex] += 1
                    
    return h


def houghCircle(tMag, dir, radiusMin, radiusMax):
    h = np.zeros((tMag.shape[0], tMag.shape[1], radiusMax - radiusMin))
    for y in range(h.shape[0]):
        for x in range(h.shape[1]):
            if tMag[y][x] > 0:
                for r in range(radiusMin, radiusMax):
                    xPos = int(x + r * math.cos(dir[y][x]))
                    yPos = int(y + r * math.sin(dir[y][x]))
                    if xPos >= 0 and xPos < h.shape[1] and yPos >= 0 and yPos < h.shape[0]:
                        h[yPos][xPos][r - radiusMin] = h[yPos][xPos][r - radiusMin] + 1

                    xPos = int(x - r * math.cos(dir[y][x]))
                    yPos = int(y - r * math.sin(dir[y][x]))
                    if xPos >= 0 and xPos < h.shape[1] and yPos >= 0 and yPos < h.shape[0]:
                        h[yPos][xPos][r - radiusMin] = h[yPos][xPos][r - radiusMin] + 1

    return h



def sobel(image, threshold = 0.5):
    image = image / 255.0
    xKrnl = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    dx = convolute(image, xKrnl)
    yKrnl = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    dy = convolute(image, yKrnl)
    mag = np.power(np.power(dx, 2) + np.power(dy, 2), 0.5)
    dir = np.arctan(dy / (dx + 0.000000001))
    tMag = (mag > threshold) * 255.0
    return tMag, dir


def printScores(groundTruth, detectionDict, threshold = 0.5):
    totalTP = 0
    totalFN = 0
    totalFP = 0
    totalF1 = 0
    totalTPR = 0
    for image_name, gt in groundTruth.items():
        detections = detectionDict[image_name]
        tp = 0
        fn = 0
        for A in gt:
            maxOverlap = 0
            for B in detections:
                maxOverlap = max(maxOverlap, intersectionOverUnion(A, B))

            if (maxOverlap >= threshold):
                tp += 1
            else:
                fn += 1
        fp = max(0,len(detections) - len(gt))
        f1Score = calculateF1(tp, fp, fn)
        if (len(gt) == 0):
            tpr = 1
        else:
            tpr = float(tp) / len(gt)

        print(image_name, "TP:", tp, "FP:", fp, "FN:", fn, "TPR:", tpr, "F1:", f1Score)

        totalTP += tp
        totalFN += fn
        totalFP += fp
        totalF1 += f1Score
        totalTPR += tpr

    num_images = len(groundTruth)
    
    print("AVERAGE: ", "TP:", totalTP / num_images, "FP:", totalFP / num_images, "FN:", totalFN / num_images, "TPR:", totalTPR / num_images,  "F1:", totalF1 / num_images)

def segmentationProbability(image, detect):
    redSum = 0
    for j in range(detect[1], detect[1] + detect[3]):
        if j < 0 or j >= image.shape[0]:
            print(j)
            continue
        for i in range(detect[0], detect[0] + detect[2]):
            if i < 0 or i >= image.shape[1]:
                continue
            redInv = image[j][i][0] + image[j][i][1]
            redSum += max(0,image[j][i][2] - redInv)
    return redSum / float(detect[2] * detect[3] * 255.0)
   
def lineProbability(sobelImage, detect, thetaResolution = 10):
    mag = sobelImage[0][detect[1]:(detect[1] + detect[3]),detect[0]:(detect[0] + detect[2])]
    dir = sobelImage[1][detect[1]:(detect[1] + detect[3]),detect[0]:(detect[0] + detect[2])]
    resolution = detect[2] * detect[3]
    width = thetaResolution
    hSpace = houghLine(mag, dir, 1, width)
    maxLine = (hSpace[0][int(hSpace.shape[1] / 2)]) * (hSpace[0][0] + hSpace[0][width - 1]) / (resolution * width)
    return maxLine

def circleProbability(sobelImage, detect):
    mag = sobelImage[0][detect[1]:(detect[1] + detect[3]),detect[0]:(detect[0] + detect[2])]
    dir = sobelImage[1][detect[1]:(detect[1] + detect[3]),detect[0]:(detect[0] + detect[2])]
    maxWidth = max(detect[2],detect[3])
    # resolution = detect[2] * detect[3]
    minRadius = 10
    maxRadius = int(maxWidth)
    hSpace = houghCircle(mag, dir, minRadius, maxRadius)
    maxCircle = np.max(hSpace[maxWidth // 2][maxWidth // 2])
    return maxCircle

def getDetections(model, cleanedImages, images):
    detectDict = {}
    print("detect")
    for image_name, image in cleanedImages.items():
        print(image_name) 
        print("Sobel")
        sobelOutput = sobel(image, threshold=0.8)
        print("Detections")
        detections = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
        filteredDetections = []
        print("Filter")
        uncleaned = images[image_name]
        for detect in detections:
            circleProb = circleProbability(sobelOutput, detect)
            print(circleProb)
            lineProb = lineProbability(sobelOutput, detect)
            print(lineProb)
            segProb = segmentationProbability(uncleaned, detect)
            print(segProb)
            print()
            if lineProb >= MinLineProbability and lineProb <= MaxLineProbability and circleProb >= MinCircleProbability and segProb >= MinSegmentProbability:
                filteredDetections.append(detect)
        
        detectDict[image_name] = filteredDetections

    return detectDict

def getGroundTruths(filename='groundtruth.txt'):
    gt = {}
    #open ground truths as txt file
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            if (line == '\n'):
                continue
            split_line = line.split(",")
            #first
            img_name = split_line[0]
            x = int(split_line[1])
            y = int(split_line[2])
            width = int(split_line[3])
            height = int(split_line[4])
            if img_name in gt:
                gt[img_name].append([x,y,width,height])
            else:
                gt[img_name] = [[x,y,width,height]]

    return gt

def displayDetections(images, detectionDict, colour, thickness = 4):
    for image_name, detections in detectionDict.items():
        for detection in detections:
            images[image_name] = cv2.rectangle(images[image_name], 
            (detection[0], detection[1]), 
            (detection[0] + detection[2], detection[1] + detection[3]), 
            colour, thickness)

def drawCircles(image, hCircle, minimumRadius = 10, threshold = 15):
    for y in range(hCircle.shape[0]):
        print(y)
        for x in range(hCircle.shape[1]):
            index = np.argmax(hCircle[y][x])
            amount = hCircle[y][x][index]
            if amount > threshold:
                radius = (index + minimumRadius)
                cv2.circle(image, (x, y), radius, (255,0,0), 2)

def drawLines(image, hLine, threshold=10):
    maxP = math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
    for p0 in range(hLine.shape[0]):
        p = maxP * float(p0) / hLine.shape[0]
        for t in range(hLine.shape[1]):
            if hLine[p0][t] > threshold:
                theta = math.pi * float(t) / hLine.shape[1] - math.pi / 2
                sint = (math.sin(theta) + 0.00001)
                c = p / sint
                dydx = -math.cos(theta) / sint
                y0 = int(c)
                y1 = int(c + dydx * image.shape[1])
                cv2.line(image, (0,y0), (image.shape[1], y1), (0,255,0), 1)

def printThresholdRanges(images, cleanImages, groundTruths):
    minCircle = 1
    maxCircle = 0
    minLine = 1
    maxLine = 0
    minSegmentation = 1
    maxSegmentation = 0
    for image_name, image in cleanImages.items():
        print(image_name)
        sobelOutput = sobel(image, threshold=0.8)
        for gt in groundTruths[image_name]:
            print("circle")
            circle = circleProbability(sobelOutput, gt)
            print(circle)
            print("line")
            line = lineProbability(sobelOutput, gt)
            print("seg")
            segmentation = segmentationProbability(images[image_name], gt)
            minCircle = min(minCircle, circle)
            maxCircle = max(maxCircle, circle)
            minLine = min(minLine, line)
            maxLine = max(maxLine, line)
            minSegmentation = min(minSegmentation, segmentation)
            maxSegmentation = max(maxSegmentation, segmentation)

    print("Parameters")
    print("line")
    print(minLine)
    print(maxLine)
    print("circle")
    print(minCircle)
    print(maxCircle)
    print("seg")
    print(minSegmentation)
    print(maxSegmentation)

#Load model
model = cv2.CascadeClassifier(ClassifierPath)
print("load images")
images = loadImages(downscale = 1)
print("clean images")
cleanImages = cleanupImages(images)

groundTruths = getGroundTruths()
# printThresholdRanges(images, cleanImages, groundTruths)
detections = getDetections(model, cleanImages, images)

    

printScores(groundTruths, detections)
displayDetections(images, detections, (0,255,0), thickness=2)
displayDetections(images, groundTruths, (0,0,255), thickness=2)

for image_name, image in images.items():
    cv2.imwrite("output/" + image_name, image)
