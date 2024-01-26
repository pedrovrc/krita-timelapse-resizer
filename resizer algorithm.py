from os import listdir
from os.path import isfile, join
from PIL import Image
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

# https://docs.opencv.org/4.8.0/db/d27/tutorial_py_table_of_contents_feature2d.html
# https://docs.opencv.org/4.8.0/d1/de0/tutorial_py_feature_homography.html

#   This function utilizes SIFT's algorithm to detect the presence of notable points
# the Train image, which are also present in the Query image.
def detectPOIs(query, train):
    grayQuery = cv.cvtColor(query, cv.COLOR_BGR2GRAY)
    grayTrain = cv.cvtColor(train, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create()
    kpQ, desQ = sift.detectAndCompute(grayQuery, None)
    kpT, desT = sift.detectAndCompute(grayTrain, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(desQ, desT, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
        
    MIN_MATCH_COUNT = 10
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kpQ[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpT[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
        h, w = grayQuery.shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        
        resultimg = cv.polylines(train, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    resultimg = cv.drawMatches(query,kpQ,train,kpT,good,None,**draw_params)
    plt.imshow(resultimg, 'gray'),plt.show()
    return [np.int32(dst)]

folderPath = "D:/Krita Recorder/20231218171411" # DEADLOCK
#folderPath = "D:/Krita Recorder/20240106230903" # RUNE
#folderPath = "D:/Krita Recorder/20240122180814" # DRUIDIC FOCUS
fileList = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]

# detection of resolution changes
flagFirst = True
changeList = []
filesInChange = []
for file in fileList:
    #print("Reading file:", file)
    currentFile = file
    im = Image.open(join(folderPath, file))
    w, h = im.size
    
    if flagFirst:
        previousFile = file
        currentW = firstW = w
        currentH = firstH = h
        flagFirst = False
        
    elif currentW != w or currentH != h:
        filesInChange.append((previousFile, currentFile))
        changeList.append((w - currentW, h - currentH))
        
        print("Change detected!")
        print("Between files:", previousFile, currentFile)
        print("Resolution change:", (w - currentW, h - currentH), "\n")
        
        currentW = w
        currentH = h
        
    previousFile = file

print("SCAN COMPLETE")
if (len(changeList) > 0):
    print("First size:", firstW, firstH)
    print("Last size:", currentW, currentH)
    print("Changes:", changeList)
    print("Files involved:", filesInChange, "\n")
else:
    print("No changes in resolution detected.")
    exit()

# load relevant images in tuples
loadedImgs = []
for (file1, file2) in filesInChange:
    loadedImgs.append((cv.imread(join(folderPath, file1)), cv.imread(join(folderPath, file2))))

# detect similarities in all relevant images and get cropping coordinates
counter = 0
coordForChanges =[]
for change in changeList:
    (trainImg, queryImg) = loadedImgs[counter]
    dst = detectPOIs(queryImg, trainImg)
    
    coordinates = []
    for list in dst:
        for elem in list:
            x, y = elem[0]
            coordinates.append((x,y))
    
    coordForChanges.append(coordinates)
    counter += 1
    
# ASSUMPTIONS:
# coordinates contains the points corresponding to the corners of the smaller image.
# coordinates is set in the following order: [top-left, bottom-left, bottom-right, top-right]
# therefore, assuming there is no considerable perspective transformation:
# coordinates[0] and [1] must have equal values of X, as must [2] and [3].
# coordinates[1] and [2] must have equal values of Y, as must [0] and [3].
# there might be slight differences due to rounding so that needs to be taken care of.
# Proposition: in case of unequal values, take average and round to integer

print(coordForChanges)