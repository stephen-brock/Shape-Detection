################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

from posixpath import split
from turtle import numinput
import numpy as np
import cv2
import os
import sys

#Relative path to classifier cascade
ClassifierPath = "NoEntryCascade/cascade.xml"

def detectAndDisplay(frame, imageName):
    imageName = imageName.split('/')[-1].split('.')[0]

	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
    # 3. Print number of Faces found
    print(len(faces))
    # 4. Draw box around faces found
    gt = readGroundtruth(imageName=imageName)
    for i in range(0, len(gt)):
        start_point = (int(gt[i][0]), int(gt[i][1]))
        end_point = (int(gt[i][0] + gt[i][2]), int(gt[i][1] + gt[i][3]))
        colour = (0, 0, 255)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

def loadImages(directory='No_entry'):
    dirs = os.listdir(directory)
    images = {}
    for dir in dirs:
        images[dir] = cv2.imread(directory + "/" + dir, 1)
    return images

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
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    return 2 * precision * recall / (precision + recall)

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
        tpr = float(tp) / len(gt)
        print(image_name, "TP:", tp, "FP:", fp, "FN:", fn, "TPR:", tpr, "F1:", f1Score)

        totalTP += tp
        totalFN += fn
        totalFP += fp
        totalF1 += f1Score
        totalTPR += tpr

    num_images = len(groundTruth)
    
    print("AVERAGE: ", "TP:", totalTP / num_images, "FP:", totalFP / num_images, "FN:", totalFN / num_images, "TPR:", totalTPR / num_images, "F1:", totalF1 / num_images)
    
                

def getDetections(model, images):
    detectDict = {}
    for img_name, image in images.items():
        detections = model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
        detectDict[img_name] = detections

    return detectDict

def getGroundTruths(filename='groundtruth.txt'):
    gt = {}
    #open ground truths as txt file
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
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

#Load model
model = cv2.CascadeClassifier(ClassifierPath)
images = loadImages()
groundTruths = getGroundTruths()
detections = getDetections(model, images)
printScores(groundTruths, detections)
displayDetections(images, detections, (0,255,0), thickness=2)
displayDetections(images, groundTruths, (0,0,255), thickness=2)

for image_name, image in images.items():
    cv2.imwrite("output/" + image_name, image)
