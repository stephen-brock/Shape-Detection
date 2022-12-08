################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

from fileinput import filename
from inspect import currentframe
from re import L
from sre_constants import MAXREPEAT
from string import whitespace
import numpy as np
import cv2
import os
import math
from random import random
import argparse

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='stop sign detection')
parser.add_argument('-name', '-n', type=str, default='all')
parser.add_argument('-segmentation', '-s', type=str, default='1')
args = parser.parse_args()

#Relative path to classifier cascade
ClassifierPath = "NoEntryCascade/cascade.xml"

#hough line parameters
LineMin = 0.005
LineMax = 0.009
LineFalloff = 0.006
LineImportance = 0.1

#hough circle parameters
CircleMin = 6
CircleMax = 100
CircleFalloff = 3
CircleImportance = 1

#segmentation red parameters
RedThreshold = 0.3
RedMin = 0.3
RedMax = 0.6
RedFalloff = 0.2
RedImportance = 1

SegmentationSearchThreshold = 0.1
SegmentationAcceptThreshold = 0.75

#segmentation white parameters
WhiteMinValue = 1.7
WhiteMaxValue = 2.0
WhiteVarInfluence = 20
WhiteVarThreshold = 0.5
WhiteFalloffValue = 0.4
WhiteMin = 0.20
WhiteMax = 0.35
WhiteFalloff = 0.075
WhiteImportance = 0.5

#acceptance threshold
TruthThreshold = 0.55

#map values between a value of 0-1 depending on variables
def probabilityFunction(x, smallest, largest, falloff):
    diff = min(1, 1 - max(smallest - x, x - largest))
    diff -= 1
    diff /= falloff

    return max(0, diff + 1) ** 2
    
#evaluate whether the filter probabilities will accept the detection or not
def evaluate(lineProb, circleProb, redProb, whiteProb):
    sum = lineProb * LineImportance + circleProb * CircleImportance + redProb * RedImportance + whiteProb * WhiteImportance
    return (sum / (LineImportance + CircleImportance + RedImportance + WhiteImportance))

#load all images from the given directory
def loadImages(directory='No_entry', downscale = 1):
    dirs = os.listdir(directory)
    images = {}
    for dir in dirs:
        image = cv2.imread(directory + "/" + dir, 1)
        images[dir] = cv2.resize(image, (int(image.shape[1] / downscale), int(image.shape[0] / downscale)))
    return images

#grayscale and normalise images
def cleanupImages(images):
    cleaned = {}
    for img_name, image in images.items():
        #grayscale
        cleaned[img_name] = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

    return cleaned

def intersection(A, B):
    #intersected box values
    xStart = max(A[0], B[0])
    xEnd = min(A[0] + A[2], B[0] + B[2])

    yStart = max(A[1], B[1])
    yEnd = min(A[1] + A[3], A[1] + A[3])
    
    dx = max(0,(xEnd - xStart))
    dy = max(0,(yEnd - yStart))
    #amount of area intersected
    return max(0, dx * dy)

#calculate proportion of intersecting boxes
#for checking if detections are correct
def intersectionOverUnion(A, B):
    areaA = A[2] * A[3]
    areaB = B[2] * B[3]
    #total area
    areaIntersection = intersection(A,B)
    # print("Intersection", areaIntersection)
    unionArea = areaA + areaB - areaIntersection
    # print("area", areaA + areaB - areaIntersection)
    return float(areaIntersection) / unionArea

#calculate f1 score
def calculateF1(TP, FP, FN):
    #avoid 0 division error
    if (TP == 0):
        return 0
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    return 2 * precision * recall / (precision + recall)

#use kernel for convolution
def useKernel(img, x, y, krnl, size):
    sum = 0
    for j in range(-size + 1, size):
        for i in range(-size + 1, size):
            yPos = (y + j)
            xPos = (x + i)
            #map within image bounds
            if xPos < 0 or yPos < 0 or xPos >= img.shape[1] or yPos >= img.shape[0]:
                continue
            mod = krnl[-j + size - 1][-i + size - 1]
            sum += img[yPos][xPos] * mod

    return sum

#convolute kernel
def convolute(img, krnl):
    newImage = np.zeros(img.shape)
    krnlSize = (krnl.shape[0] // 2) + 1
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            newImage[y][x] = useKernel(img, x, y, krnl, krnlSize)

    return newImage

#calculate hough line space
def houghLine(thresholdMag, dir, pResolution, thetaResolution):
    #calculate maximum distance from origin
    maxP = math.sqrt(thresholdMag.shape[0] ** 2 + thresholdMag.shape[1] ** 2)
    h = np.zeros((pResolution, thetaResolution))

    for y in range(thresholdMag.shape[0]):
        for x in range(thresholdMag.shape[1]):
            d = dir[y][x]
            if thresholdMag[y][x] > 0:
                #calculate index based on theta value and resolution
                thetaIndex = int(thetaResolution * (d + math.pi / 2) / math.pi)
                p = x * math.cos(d) + y * math.sin(d)
                if thetaIndex >= 0 and thetaIndex < thetaResolution:
                    h[int(pResolution * p / maxP)][thetaIndex] += 1
                    
    return h

#calculate hough circle space
def houghCircle(tMag, dir, radiusMin, radiusMax):
    h = np.zeros((tMag.shape[0], tMag.shape[1], radiusMax - radiusMin))
    for y in range(h.shape[0]):
        for x in range(h.shape[1]):
            if tMag[y][x] > 0:
                for r in range(radiusMin, radiusMax):
                    #calculate origin for given radius
                    cosR = r * math.cos(dir[y][x])
                    sinR = r * math.sin(dir[y][x])
                    xPos = int(x + cosR)
                    yPos = int(y + sinR)
                    #test whether the origin is within the image
                    if xPos >= 0 and xPos < h.shape[1] and yPos >= 0 and yPos < h.shape[0]:
                        h[yPos][xPos][r - radiusMin] = h[yPos][xPos][r - radiusMin] + 1

                    #calculate opposite origin
                    xPos = int(x - cosR)
                    yPos = int(y - sinR)
                    #test whether the origin is within the image
                    if xPos >= 0 and xPos < h.shape[1] and yPos >= 0 and yPos < h.shape[0]:
                        h[yPos][xPos][r - radiusMin] = h[yPos][xPos][r - radiusMin] + 1

    return h


#calculate sobel image
def sobel(image, threshold = 0.5):
    #map pixels between 0-1
    image = image / 255.0
    #convolute kernel
    xKrnl = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    dx = convolute(image, xKrnl)
    yKrnl = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    dy = convolute(image, yKrnl)
    #calculate magnitude
    mag = np.power(np.power(dx, 2) + np.power(dy, 2), 0.5)
    #calculate direction
    dir = np.arctan(dy / (dx + 0.000000001))
    #threshold magnitude
    tMag = (mag > threshold) * 255.0
    return tMag, dir

#display scores relative to ground truth
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
                #record max overlap to avoid duplicate true positives
                maxOverlap = max(maxOverlap, intersectionOverUnion(A, B))

            #if overlap greater than a threshold then true positive
            if (maxOverlap >= threshold):
                tp += 1
            else:
                #otherwise false negative
                fn += 1
        #false positive is number of detections which are not true
        fp = max(0,len(detections) - tp)
        f1Score = calculateF1(tp, fp, fn)
        #if negative image (no ground truths) assume 100% true positive rate
        if (len(gt) == 0):
            tpr = 1
        else:
            tpr = float(tp) / (tp + fn)

        print(image_name, "TP:", tp, "FP:", fp, "FN:", fn, "TPR:", tpr, "F1:", f1Score)

        totalTP += tp
        totalFN += fn
        totalFP += fp
        totalF1 += f1Score
        totalTPR += tpr

    num_images = len(groundTruth)
    
    print("AVERAGE: ", "TP:", totalTP / num_images, "FP:", totalFP / num_images, "FN:", totalFN / num_images, "TPR:", totalTPR / num_images,  "F1:", totalF1 / num_images)

#if pixel is red
def isRed(col, average):
    #if difference between red channel and others are above a given threshold
    return ((int(col[2]) - int(col[0]) - int(col[1])) / average) > RedThreshold

#is pixel white compared to average
def isWhite(col, cleanCol, average):
    #if sum of the colours in relation to average 
    avg = col / average
    avgWhite = cleanCol / average
    return (probabilityFunction(avgWhite, WhiteMinValue, WhiteMaxValue, WhiteFalloffValue) - WhiteVarInfluence * np.var(avg)) > WhiteVarThreshold


def createRedImage(image, average):
    redImage = np.zeros((image.shape[0],image.shape[1]))
    for i in range(redImage.shape[0]):
        for j in range(redImage.shape[1]):
            redImage[i][j] = isRed(image[i][j], average)
    return redImage

def createWhiteImage(image, cleanImage, average):
    whiteImage = np.zeros((image.shape[0],image.shape[1]))
    for i in range(whiteImage.shape[0]):
        for j in range(whiteImage.shape[1]):
            whiteImage[i][j] = isWhite(image[i][j], cleanImage[i][j], average)
    return whiteImage
    
#linearly biases the center of detection
#multiplied by 2 for normalisation
def centerBias(pos, center, width):
    return 2 * max(0, 1 - 2 * abs(pos - center) / width)

#probability depending on the colours in the image
def segmentationProbability(image, clean, average, detect):
    redSum = 0
    whiteSum = 0
    for j in range(detect[1], detect[1] + detect[3]):
        if j < 0 or j >= image.shape[0]:
            continue
        for i in range(detect[0], detect[0] + detect[2]):
            if i < 0 or i >= image.shape[1]:
                continue
            col = image[j][i]
            cleanCol = clean[j][i]
            #sum pixels which pass red or white test
            redSum += isRed(col, average)
            whiteSum += isWhite(col, cleanCol, average) * centerBias(j, detect[1] + detect[3] / 2, detect[3]) * centerBias(i, detect[0] + detect[2] / 2, detect[2])
    sum = float(detect[2] * detect[3])
    #divide by the size of the detection
    #for a proportion of pixels
    return redSum / sum, whiteSum / sum

def segmentationProbabilityCached(redImage, whiteImage, detect):
    redSum = 0
    whiteSum = 0
    for j in range(detect[1], detect[1] + detect[3]):
        if j < 0 or j >= redImage.shape[0]:
            continue
        for i in range(detect[0], detect[0] + detect[2]):
            if i < 0 or i >= redImage.shape[1]:
                continue
            #sum pixels which pass red or white test
            redSum += redImage[j][i]
            whiteSum += whiteImage[j][i] * centerBias(j, detect[1] + detect[3] / 2, detect[3]) * centerBias(i, detect[0] + detect[2] / 2, detect[2])
    sum = float(detect[2] * detect[3])
    #divide by the size of the detection
    #for a proportion of pixels
    return redSum / sum, whiteSum / sum

def segmentationProbabilityBiased(redImage, detect):
    redSum = 0
    for j in range(detect[1], detect[1] + detect[3]):
        if j < 0 or j >= redImage.shape[0]:
            continue
        for i in range(detect[0], detect[0] + detect[2]):
            if i < 0 or i >= redImage.shape[1]:
                continue
            #sum pixels which pass red or white test
            # bias = centerBias(i, detect[0] + detect[2] / 2, detect[2]) * centerBias(j, detect[1] + detect[3] / 2, detect[3])
            redSum += redImage[j][i]
    sum = float(detect[2] * detect[3])
    #divide by the size of the detection
    #for a proportion of pixels
    return redSum / sum

#return value corresponding of vertical and horizontal lines 
#negative padding to ideally focus on the rectangle inside of stop sign rather than exterior features
def lineProbability(sobelImage, detect, thetaResolution = 5, padding = -5):
    #pad detection
    left = max(0, detect[1] - padding)
    right = min(sobelImage[0].shape[0] - 1, detect[1] + detect[3] + padding)
    top = max(0, detect[0] - padding)
    bottom = min(sobelImage[0].shape[1] - 1, detect[0] + detect[2] + padding)

    mag = sobelImage[0][left:right,top:bottom]
    dir = sobelImage[1][left:right,top:bottom]
    resolution = detect[2] * detect[3]
    #smaller the theta resolution the more leeway in the angles
    width = thetaResolution
    hSpace = houghLine(mag, dir, 1, width)
    #line amount in relation to resolution
    maxLine = (hSpace[0][int(hSpace.shape[1] / 2)] + hSpace[0][0] + hSpace[0][width - 1]) / (resolution * width * 3)
    return maxLine

#return the maximum value for any radius in the center of the hough space
def circleProbability(sobelImage, detect, padding = 5):
    #pad detection
    left = max(0, detect[1] - padding)
    right = min(sobelImage[0].shape[0] - 1, detect[1] + detect[3] + padding)
    top = max(0, detect[0] - padding)
    bottom = min(sobelImage[0].shape[1] - 1, detect[0] + detect[2] + padding)

    #create subset based on detection
    mag = sobelImage[0][left:right,top:bottom]
    dir = sobelImage[1][left:right,top:bottom]

    maxWidth = max(bottom - top, right - left)
    minRadius = 10
    #assume radius is not larger than the image (accounted for in padding)
    maxRadius = int(maxWidth + padding)

    #calculate hough space
    hSpace = houghCircle(mag, dir, minRadius, maxRadius)
    #maximum radius for center of hough space (assume detection is centered)
    maxCircle = np.max(hSpace[(bottom - top) // 2][(right - left) // 2])
    return maxCircle

#get detections using the cascade model and filter them with viola-jones/colour analysis
def getDetections(model, cleanedImages, images, extraDetections = {}):
    print("detect")
    detectDict = {}
    for image_name, image in cleanedImages.items():
        print(image_name) 
        sobelOutput = sobel(image, threshold=0.5)
        average = np.median(image)
        detections = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
        detections = list(detections)
        filteredDetections = []
        scores = []
        uncleaned = images[image_name]
        if image_name in extraDetections:
            detections.extend(extraDetections[image_name])

        #begin filtering detections
        for detect in detections:
            print()
            redSegProb, whiteSegProb = segmentationProbability(uncleaned, image, average, detect)
            pSegRed = probabilityFunction(redSegProb, RedMin, RedMax, RedFalloff)
            pSegWhite = probabilityFunction(whiteSegProb, WhiteMin, WhiteMax, WhiteFalloff)
            lineProb = lineProbability(sobelOutput, detect)
            pLine = probabilityFunction(lineProb, LineMin, LineMax, LineFalloff)
            circleProb = circleProbability(sobelOutput, detect)
            pCircle = probabilityFunction(circleProb, CircleMin, CircleMax, CircleFalloff)
            score = evaluate(pLine, pCircle, pSegRed, pSegWhite) 
            #evaluate probabilities
            if score >= TruthThreshold:
                #accept detection
                filteredDetections.append(detect)
                scores.append(score)
        
        filteredDetections = removeDuplicates(filteredDetections, scores)
        detectDict[image_name] = filteredDetections

    return detectDict

def removeDuplicates(detections, scores):
    newDetections = []
    copies = []
    while len(detections) > 0:
        copies.append(0)
        for j in range(1, len(detections)):
            if (intersectionOverUnion(detections[0], detections[j]) > 0.1):
                copies.append(j)
        
        bestProb = -1
        bestDetect = -1
        for copy in copies:
            if scores[copy] > bestProb:
                bestProb = scores[copy]
                bestDetect = copy
        
        newDetections.append(detections[bestDetect])

        offset = 0
        for copy in copies:
            detections.pop(copy - offset)
            scores.pop(copy - offset)
            offset += 1
        
        copies[:] = []
    return newDetections

#partition the segmentation space to reduce duplicates and
#increase efficiency by only seraching in areas with red space
def partitionDetections(redImage, whiteImage, maxSpacing, maxSize, padding):
    #size equal to half size of image, spliting into around 4 parts
    currentSize = maxSize
    #first detection is the whole image
    searchDetections = [(0,0,redImage.shape[1] - 1, redImage.shape[0] - 1)]
    detections = []
    newSearch = []
    newSearchDetections = []
    while currentSize > maxSpacing and len(searchDetections) > 0:
        print("partition size", currentSize)
        #quad tree therefore half size on each iteration to subdivide the space
        currentSize = currentSize // 2
        for currentDetect in searchDetections:
            #amount of iterations to fit inside the image (2 for every iteration other than first)
            xAmount = int(currentDetect[2] // currentSize)
            yAmount = int(currentDetect[3] // currentSize)
            for j in range(yAmount):
                #stop looking this branch if an appropriate detection has been found
                # if found:
                #     break;
                for i in range(xAmount):
                    #create detection
                    detect = (currentDetect[0] + i * currentSize, currentDetect[1] + j * currentSize, currentSize, currentSize)
                    #test detection
                    redSegProb = segmentationProbabilityBiased(redImage, detect)

                    #possible detection
                    if (redSegProb > SegmentationSearchThreshold):
                        print("end search", detect)
                        #no more searching under this branch
                        #find best detection within this branch and use it as partition
                        width = int(currentDetect[2] * padding)
                        height = int(currentDetect[3] * padding)
                        x = max(0, detect[0] - width // 2)
                        y = max(0, detect[1] - height // 2)
                        #constrain to image borders
                        width = width - max(0, 1 + x + width - redImage.shape[1])
                        height = height - max(0, 1 + y + height - redImage.shape[0])
                        detections.append((x, y, width, height))
                        # break
                    elif (redSegProb > 0):
                        #this branch has possible detection, keep looking
                        print("search", detect)
                        newSearch.append(detect)
            
            newSearchDetections.extend(newSearch)
            newSearch[:] = []
        #new searches
        searchDetections = list(newSearchDetections)
        newSearchDetections[:] = []
    return detections

#gets detections based on segmentation
def getDetectionsProbability(cleanedImages, images, minSize, maxSize, scaleFactor, maxSpacing, iterations):
    detectDict = {}
    print("detect")
    for image_name, image in cleanedImages.items():
        print(image_name) 
        uncleaned = images[image_name]
        #partions space into areas with segmentation and without
        #reduces duplicates and increases performance
        average = np.median(image)
        redImage = createRedImage(uncleaned, average)
        whiteImage = createWhiteImage(uncleaned, image, average)
        partitioned = partitionDetections(redImage, whiteImage, maxSpacing, maxSize, 1.5)
        detections = []
        scores = []
        for partition in partitioned:
            #records best detection
            bestList = [partition]
            #size cannot be larger than maximum given size
            currentSize = int(min(partition[2], partition[3], maxSize) / scaleFactor)
            # sizeIncrement = int(currentSize * (scaleFactor - 1))
            # currentSize -= sizeIncrement
            highestProb = -1
            nextPartition = partition
            while currentSize >= minSize and highestProb < RedMax:
                print("iter size",currentSize)
                xSpacing = (nextPartition[2] - currentSize) // (iterations - 1)
                ySpacing = (nextPartition[3] - currentSize) // (iterations - 1)
                #iterate for iterations^2
                for j in range(iterations):
                    for i in range(iterations):
                        #calculate detection position and scale
                        detect = (nextPartition[0] + i * xSpacing, nextPartition[1] + j * ySpacing, currentSize, currentSize)
                        redSegProb, whiteSegProb = segmentationProbabilityCached(redImage, whiteImage, detect)
                        # uncleaned = cv2.rectangle(uncleaned, (int(detect[0]), int(detect[1])), (int(detect[0] + detect[2]), int(detect[1] + detect[3])), (int(random() * 255),int(random() * 255),int(random() * 255)), 1)

                        #find maximum probability
                        if (redSegProb >= highestProb):
                            highestProb = redSegProb
                            nextPartition = detect
                            bestList.append(detect)
                # currentSize -= sizeIncrement
                currentSize = int(currentSize / scaleFactor)
            #if maximum probability above the threshold append to detections
            highestProb = -1
            bestDetect = None
            for detect in bestList:
                redSegProb, whiteSegProb = segmentationProbabilityCached(redImage, whiteImage, detect)
                prob = (probabilityFunction(redSegProb, RedMin, RedMax, RedFalloff) * RedImportance + probabilityFunction(whiteSegProb, WhiteMin, WhiteMin, WhiteFalloff) * WhiteImportance) / (RedImportance + WhiteImportance)
                if prob > highestProb:
                    highestProb = prob
                    bestDetect = detect
            if highestProb > SegmentationAcceptThreshold:
                print("DETECTED", bestDetect)
                detections.append(bestDetect)
                scores.append(highestProb)
        
        #remove any duplicates/overlapping detections
        #cause by large padding resulting in overlapping partitions
        detections = removeDuplicates(detections, scores)
        # for detect in detections:
        #     uncleaned = cv2.rectangle(uncleaned, (int(detect[0]), int(detect[1])), (int(detect[0] + detect[2]), int(detect[1] + detect[3])), (0,255,0), 1)

        # for partition in partitioned:
        #     uncleaned = cv2.rectangle(uncleaned, (int(partition[0]), int(partition[1])), (int(partition[0] + partition[2]), int(partition[1] + partition[3])), (255,0,0), 1)
        # cv2.imwrite("output/partition/" + image_name, uncleaned)
        detectDict[image_name] = detections

    return detectDict

#read ground truths from text file
def getGroundTruths(filename='groundtruth.txt'):
    gt = {}
    #open ground truths as txt file
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            if (line == '\n'):
                continue
            split_line = line.split(",")
            #name
            img_name = split_line[0]
            #detection
            x = int(split_line[1])
            y = int(split_line[2])
            width = int(split_line[3])
            height = int(split_line[4])
            if img_name in gt:
                #append as another detection
                gt[img_name].append([x,y,width,height])
            else:
                #new file name referenced
                gt[img_name] = [[x,y,width,height]]

    return gt

#draw rectangles indicating detections on image
def displayDetections(images, detectionDict, colour, thickness = 4):
    for image_name, detections in detectionDict.items():
        for detection in detections:
            images[image_name] = cv2.rectangle(images[image_name], 
            (detection[0], detection[1]), 
            (detection[0] + detection[2], detection[1] + detection[3]), 
            colour, thickness)

#draw the circles generated by hough circle space
def drawCircles(image, hCircle, minimumRadius = 10, threshold = 15):
    for y in range(hCircle.shape[0]):
        print(y)
        for x in range(hCircle.shape[1]):
            index = np.argmax(hCircle[y][x])
            amount = hCircle[y][x][index]
            if amount > threshold:
                radius = (index + minimumRadius)
                cv2.circle(image, (x, y), radius, (255,0,0), 2)

#draw the lines generated by hough line space
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

#filters detections by true positive and false positive
def removeIncorrectDetections(detections, groundTruths):
    correct = []
    incorrect = []
    for A in detections:
        maxOverlap = 0
        for B in groundTruths:
            maxOverlap = max(maxOverlap, intersectionOverUnion(A, B))

        #true if detection is accepted
        if (maxOverlap >= 0.5):
            correct.append(A)
        else:
            incorrect.append(A)
    return correct, incorrect

#writes outputs in a csv table for analysis
def writeOutputsToFile(redSeg, redSegFalse, whiteSeg, whiteSegFalse, line, lineFalse, circle, circleFalse, filename):
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

def writeOutputs(images, cleanImages, groundTruths, detectionDict, filename="output.csv"):
    #for storing outputs of true and false positives
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
        average = np.median(image)
        sobelOutput = sobel(image, threshold=0.5)
        #filter by true positive/false positive
        detections = detectionDict[image_name]
        correct, incorrect = removeIncorrectDetections(detections, groundTruths[image_name])
        for gt in correct:
            c = circleProbability(sobelOutput, gt)
            circle.append(c)
            l = lineProbability(sobelOutput, gt)
            line.append(l)
            redSegment, whiteSegment = segmentationProbability(images[image_name], image, average, gt)
            redSeg.append(redSegment)
            whiteSeg.append(whiteSegment)
        for f in incorrect:
            c = circleProbability(sobelOutput, f)
            circleFalse.append(c)
            l = lineProbability(sobelOutput, f)
            lineFalse.append(l)
            redSegment, whiteSegment = segmentationProbability(images[image_name], image, average, f)
            redSegFalse.append(redSegment)
            whiteSegFalse.append(whiteSegment)
    
    #write outputs to csv
    writeOutputsToFile(redSeg, redSegFalse, whiteSeg, whiteSegFalse, line, lineFalse, circle, circleFalse, filename)
        
#print variable data (e.g. minimum, maximum)
def printThresholdRanges(images, cleanImages, groundTruths):
    #for storing outputs
    redSeg = []
    whiteSeg = []
    line = []
    circle = []
    segmentDetections = getDetectionsProbability(cleanImages, images, 10, 300, 1.1, 30, 2)
    for image_name, image in cleanImages.items():
        print(image_name)
        average = np.mean(image)
        sobelOutput = sobel(image, threshold=0.5) 
        detections = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
        detections = list(detections)
        if image_name in segmentDetections:
            detections.extend(segmentDetections[image_name])
        #filter detections for true positive/false positive
        correct, incorrect = removeIncorrectDetections(detections, groundTruths[image_name])
        for gt in correct:
            # calculate probabilities for true positive
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
            #true if true positive was rejected by filter
            score = evaluate(pLine, pCircle, pSegRed, pSegWhite)
            if (not score >= TruthThreshold):
                print("REJECTED", score, l, c, redSegment, whiteSegment)
        for f in incorrect:
            # calculate probabilities for false positive
            c = circleProbability(sobelOutput, f)
            l = lineProbability(sobelOutput, f)
            redSegment, whiteSegment = segmentationProbability(images[image_name], image, average, f)

            pSegRed = probabilityFunction(redSegment, RedMin, RedMax, RedFalloff)
            pSegWhite = probabilityFunction(whiteSegment, WhiteMin, WhiteMax, WhiteFalloff)
            pLine = probabilityFunction(l, LineMin, LineMax, LineFalloff)
            pCircle = probabilityFunction(c, CircleMin, CircleMax, CircleFalloff)
            #true if false positive was accepted by filter
            score = evaluate(pLine, pCircle, pSegRed, pSegWhite)
            if (score >= TruthThreshold):
                print("ACCEPTED", score, l, c, redSegment, whiteSegment)
    
    #print parameters
    print("Parameters")
    # print("line")
    # print(np.min(line),  np.max(line))
    # print("circle")
    # print(np.min(circle), np.max(circle))
    print("seg")
    print(np.min(redSeg), np.max(redSeg))
    print("seg white")
    print(np.min(whiteSeg), np.max(whiteSeg))

#write hough spaces as images
def writeHoughSpaces(cleanImages):
    for image_name, image in cleanImages.items():
        print(image_name)
        sobel_image = sobel(image)
        print("line")
        hLine = houghLine(sobel_image[0], sobel_image[1], 200, 200) * 2
        print("circle")
        #radius range between 3-150
        hCircle = houghCircle(sobel_image[0], sobel_image[1], 10, 300)
        circleImage = np.zeros((hCircle.shape[0], hCircle.shape[1]))
        for y in range(hCircle.shape[0]):
            for x in range(hCircle.shape[1]):
                #threshold if a radius is above the minimum value
                circleImage[y][x] = (np.max(hCircle[y][x]) > CircleMin) * 255.0

        cv2.imwrite("output/line/" + image_name, hLine)
        cv2.imwrite("output/circle/" + image_name, circleImage)

#Load model
model = cv2.CascadeClassifier(ClassifierPath)

if (args.name == 'all'):
    print("load images")
    #load images into list
    images = loadImages(downscale = 1)
    print("clean images")
    #normalise image into one channel
    cleanImages = cleanupImages(images)
    segmentDetections = {}

    if (args.segmentation == '1'):
        #gets detections based on segmentation
        segmentDetections = getDetectionsProbability(cleanImages, images, 10, 300, 1.1, 30, 2)
    
    #calculates and filters detections
    detections = getDetections(model, cleanImages, images, segmentDetections)
    
    #read ground truths from text file
    groundTruths = getGroundTruths()
    #prints data about variables
    # printThresholdRanges(images, cleanImages, groundTruths)
    # writes all value outputs to csv file for analysis
    # writeOutputs(images, cleanImages, groundTruths, detections, filename="output2.csv")

    #score detections compared to ground truths
    printScores(groundTruths, detections)
    # #draw detections
    displayDetections(images, detections, (0,255,0), thickness=2)
    # #draw ground truths
    displayDetections(images, groundTruths, (0,0,255), thickness=2)

    #write images with drawn detections
    for image_name, image in images.items():
        print(image_name)
        cv2.imwrite("output/" + image_name, image)
else:
    images = {}
    images[args.name] = cv2.imread(args.name, 1)
    print("clean images")
    #normalise image into one channel
    cleanImages = cleanupImages(images)
    #prints data about variables
    # printThresholdRanges(images, cleanImages, groundTruths)
    #gets detections based on segmentation
    segmentDetections = {}

    if (args.segmentation == '1'):
        #gets detections based on segmentation
        segmentDetections = getDetectionsProbability(cleanImages, images, 10, 300, 1.1, 30, 2)
    
    #calculates and filters detections
    detections = getDetections(model, cleanImages, images, segmentDetections)
    #read ground truths from text file
    groundTruths = getGroundTruths()
    #draw detections
    displayDetections(images, detections, (0,255,0), thickness=2)

    #write images with drawn detections
    for image_name, image in images.items():
        cv2.imwrite("detected.bmp", image)