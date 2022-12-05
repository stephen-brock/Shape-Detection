################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

from fileinput import filename
from sre_constants import MAXREPEAT
from string import whitespace
import numpy as np
import cv2
import os
import math

#Relative path to classifier cascade
ClassifierPath = "NoEntryCascade/cascade.xml"

MinProbability = 0.05
MaxProbability = 0.95

LineMin = 2
LineMax = 14
LineFalloff = 50
LineImportance = 0.1

CircleMin = 6
CircleMax = 100
CircleFalloff = 7
CircleImportance = 1

RedThreshold = 60
RedMin = 0.3
RedMax = 0.6
RedFalloff = 0.5
RedImportance = 1

WhiteThreshold = 1.5
WhiteMin = 0.28
WhiteMax = 0.3
WhiteFalloff = 0.3
WhiteImportance = 0.2

TruthThreshold = 0.55

def probabilityFunction(x, smallest, largest, falloff):
    diff = min(1, 1 - max(smallest - x, x - largest))
    diff = min(1, diff) - 1
    diff /= falloff

    return max(MinProbability, min(MaxProbability, max(0, (diff + 1) ** 2)))

# def evaluate(lineProb, circleProb):
#     sum = lineProb * LineImportance + circleProb * CircleImportance
#     return (sum / (LineImportance + CircleImportance)) >= TruthThreshold

def evaluate(lineProb, circleProb, redProb, whiteProb):
    sum = lineProb * LineImportance + circleProb * CircleImportance + redProb * RedImportance + whiteProb * WhiteImportance
    return (sum / (LineImportance + CircleImportance + RedImportance + WhiteImportance)) >= TruthThreshold

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
        #grayscale
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

def isRed(col):
    return (int(col[2]) - int(col[0]) - int(col[1])) > RedThreshold

def isWhite(col, average):
    return (np.sum(col) / average) > WhiteThreshold

def segmentationProbability(image, clean, average, detect):
    redSum = 0
    whiteSum = 0
    for j in range(detect[1], detect[1] + detect[3]):
        if j < 0 or j >= image.shape[0]:
            print(j)
            continue
        for i in range(detect[0], detect[0] + detect[2]):
            if i < 0 or i >= image.shape[1]:
                continue
            col = image[j][i]
            cleanCol = clean[j][i]
            redSum += isRed(col)
            whiteSum += isWhite(cleanCol, average)
    sum = float(detect[2] * detect[3])
    return redSum / sum, whiteSum / sum
   
def lineProbability(sobelImage, detect, thetaResolution = 5, padding = -5):
    left = max(0, detect[1] - padding)
    right = min(sobelImage[0].shape[0] - 1, detect[1] + detect[3] + padding)
    top = max(0, detect[0] - padding)
    bottom = min(sobelImage[0].shape[1] - 1, detect[0] + detect[2] + padding)

    mag = sobelImage[0][left:right,top:bottom]
    dir = sobelImage[1][left:right,top:bottom]
    resolution = detect[2] * detect[3]
    width = thetaResolution
    hSpace = houghLine(mag, dir, 1, width)
    maxLine = (hSpace[0][int(hSpace.shape[1] / 2)]) * (hSpace[0][0] + hSpace[0][width - 1]) / (resolution * width)
    return maxLine

def circleProbability(sobelImage, detect, padding = 5):
    left = max(0, detect[1] - padding)
    right = min(sobelImage[0].shape[0] - 1, detect[1] + detect[3] + padding)
    top = max(0, detect[0] - padding)
    bottom = min(sobelImage[0].shape[1] - 1, detect[0] + detect[2] + padding)

    mag = sobelImage[0][left:right,top:bottom]
    dir = sobelImage[1][left:right,top:bottom]
    maxWidth = max(bottom - top, right - left)
    minRadius = 10
    maxRadius = int(maxWidth + padding)
    hSpace = houghCircle(mag, dir, minRadius, maxRadius)
    maxCircle = np.max(hSpace[(bottom - top) // 2][(right - left) // 2])
    return maxCircle

def getDetections(model, cleanedImages, images, threshold = 0.5):
    detectDict = {}
    print("detect")
    for image_name, image in cleanedImages.items():
        print(image_name) 
        average = np.average(image)
        sobelOutput = sobel(image, threshold=0.5)
        detections = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
        filteredDetections = []
        uncleaned = images[image_name]
        for detect in detections:
            print()
            # redSegProb, whiteSegProb = segmentationProbability(uncleaned, image, average, detect)
            # pSegRed = probabilityFunction(redSegProb, RedMin, RedMax, RedFalloff)
            # pSegWhite = probabilityFunction(whiteSegProb, WhiteMin, WhiteMax, WhiteFalloff)
            lineProb = lineProbability(sobelOutput, detect)
            pLine = probabilityFunction(lineProb, LineMin, LineMax, LineFalloff)
            circleProb = circleProbability(sobelOutput, detect)
            pCircle = probabilityFunction(circleProb, CircleMin, CircleMax, CircleFalloff)
            if (evaluate(pLine, pCircle)):
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

def removeIncorrectDetections(detections, groundTruths):
    correct = []
    incorrect = []
    for A in detections:
        maxOverlap = 0
        for B in groundTruths:
            maxOverlap = max(maxOverlap, intersectionOverUnion(A, B))

        if (maxOverlap >= 0.5):
            correct.append(A)
        else:
            incorrect.append(A)
    return correct, incorrect

def writeOutputs(images, cleanImages, groundTruths, filename="output.csv"):
    redSeg = []
    redSegFalse = []
    whiteSeg = []
    whiteSegFalse = []
    line = []
    lineFalse = []
    circle = []
    circleFalse = []
    for image_name, image in cleanImages.items():
        print(image_name)
        average = np.mean(image)
        sobelOutput = sobel(image, threshold=0.5) 
        detections = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
        correct, incorrect = removeIncorrectDetections(detections, groundTruths[image_name])
        for gt in correct:
            c = circleProbability(sobelOutput, gt)
            circle.append(c)
            l = lineProbability(sobelOutput, gt)
            line.append(l)
            redSegment, whiteSegment = segmentationProbability(images[image_name], image, average, gt)
            redSeg.append(redSegment)
            whiteSeg.append(whiteSegment)
            
            pSegRed = probabilityFunction(redSegment, RedMin, RedMax, RedFalloff)
            pSegWhite = probabilityFunction(whiteSegment, WhiteMin, WhiteMax, WhiteFalloff)
            pLine = probabilityFunction(l, LineMin, LineMax, LineFalloff)
            pCircle = probabilityFunction(c, CircleMin, CircleMax, CircleFalloff)
            if (not evaluate(pLine, pCircle, pSegRed, pSegWhite)):
                print("REJECTED", l, c, redSegment, whiteSegment)
        for f in incorrect:
            c = circleProbability(sobelOutput, f)
            circleFalse.append(c)
            l = lineProbability(sobelOutput, f)
            lineFalse.append(l)
            redSegment, whiteSegment = segmentationProbability(images[image_name], image, average, f)
            redSegFalse.append(redSegment)
            whiteSegFalse.append(whiteSegment)
            
            pSegRed = probabilityFunction(redSegment, RedMin, RedMax, RedFalloff)
            pSegWhite = probabilityFunction(whiteSegment, WhiteMin, WhiteMax, WhiteFalloff)
            pLine = probabilityFunction(l, LineMin, LineMax, LineFalloff)
            pCircle = probabilityFunction(c, CircleMin, CircleMax, CircleFalloff)
            if (evaluate(pLine, pCircle, pSegRed, pSegWhite)):
                print("ACCEPTED", l, c, redSegment, whiteSegment)
    
    with open(filename, 'w') as f:
        f.write("Red Segment\n")
        f.write("True")
        for i in redSeg:
            f.write("," + str(i))
        f.write("\nFalse")
        for i in redSegFalse:
            f.write("," + str(i))

        f.write("\nWhite Segment\n")
        f.write("True")
        for i in whiteSeg:
            f.write("," + str(i))
        f.write("\nFalse")
        for i in whiteSegFalse:
            f.write("," + str(i))

        f.write("\nLine\n")
        f.write("True")
        for i in line:
            f.write("," + str(i))
        f.write("\nFalse")
        for i in lineFalse:
            f.write("," + str(i))

        f.write("\nCircle\n")
        f.write("True")
        for i in circle:
            f.write("," + str(i))
        f.write("\nFalse")
        for i in circleFalse:
            f.write("," + str(i))
        

def printThresholdRanges(images, cleanImages, groundTruths):
    redSeg = []
    whiteSeg = []
    line = []
    circle = []
    for image_name, image in cleanImages.items():
        print(image_name)
        average = np.mean(image)
        sobelOutput = sobel(image, threshold=0.5) 
        detections = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
        correct, incorrect = removeIncorrectDetections(detections, groundTruths[image_name])
        for gt in correct:
            c = circleProbability(sobelOutput, gt)
            circle.append(c)
            l = lineProbability(sobelOutput, gt)
            line.append(l)
            redSegment, whiteSegment = segmentationProbability(images[image_name], image, average, gt)
            redSeg.append(redSegment)
            whiteSeg.append(whiteSegment)
            
            pSegRed = probabilityFunction(redSegment, RedMin, RedMax, RedFalloff)
            pSegWhite = probabilityFunction(whiteSegment, WhiteMin, WhiteMax, WhiteFalloff)
            pLine = probabilityFunction(l, LineMin, LineMax, LineFalloff)
            pCircle = probabilityFunction(c, CircleMin, CircleMax, CircleFalloff)
            if (not evaluate(pLine, pCircle, pSegRed, pSegWhite)):
                print("REJECTED", l, c, redSegment, whiteSegment)
        for f in incorrect:
            c = circleProbability(sobelOutput, f)
            l = lineProbability(sobelOutput, f)
            redSegment, whiteSegment = segmentationProbability(images[image_name], image, average, f)

            pSegRed = probabilityFunction(redSegment, RedMin, RedMax, RedFalloff)
            pSegWhite = probabilityFunction(whiteSegment, WhiteMin, WhiteMax, WhiteFalloff)
            pLine = probabilityFunction(l, LineMin, LineMax, LineFalloff)
            pCircle = probabilityFunction(c, CircleMin, CircleMax, CircleFalloff)
            if (evaluate(pLine, pCircle, pSegRed, pSegWhite)):
                print("ACCEPTED", l, c, redSegment, whiteSegment)
    
    print("Parameters")
    print("line")
    print(np.min(line),  np.max(line))
    print("circle")
    print(np.min(circle), np.max(circle))
    print("seg")
    print(np.min(redSeg), np.max(redSeg))
    print("seg white")
    print(np.min(whiteSeg), np.max(whiteSeg))

def printHoughSpaces(cleanImages):
    for image_name, image in cleanImages.items():
        print(image_name)
        sobel_image = sobel(image)
        print("line")
        hLine = houghLine(sobel_image[0], sobel_image[1], 200, 200) * 2
        print("circle")
        hCircle = houghCircle(sobel_image[0], sobel_image[1], 10, 300)
        circleImage = np.zeros((hCircle.shape[0], hCircle.shape[1]))
        for y in range(hCircle.shape[0]):
            for x in range(hCircle.shape[1]):
                circleImage[y][x] = (np.max(hCircle[y][x]) > MinCircleProbability) * 255.0

        cv2.imwrite("output/line/" + image_name, hLine)
        cv2.imwrite("output/circle/" + image_name, circleImage)

#Load model
model = cv2.CascadeClassifier(ClassifierPath)
print("load images")
images = loadImages(downscale = 1)
print("clean images")
cleanImages = cleanupImages(images)
groundTruths = getGroundTruths()
# writeOutputs(images, cleanImages, groundTruths)
# printThresholdRanges(images, cleanImages, groundTruths)
detections = getDetections(model, cleanImages, images)

printScores(groundTruths, detections)
displayDetections(images, detections, (0,255,0), thickness=2)
displayDetections(images, groundTruths, (0,0,255), thickness=2)

for image_name, image in images.items():
    cv2.imwrite("output/" + image_name, image)
