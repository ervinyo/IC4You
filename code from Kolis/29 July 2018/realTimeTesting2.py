import os

import matplotlib

matplotlib.use('AGG')

from keras.models import model_from_json
from keras.losses import categorical_crossentropy
import videoto3d

import numpy as np
import pyrealsense2 as rs
import cv2
import csv
import time

handin = False

initialState = 1
currentState = initialState
nextState = -1


def loadFSM(filename):
    fsmTable = []

    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            fsmTable.append(row)

    npFsmTable = np.asarray(fsmTable, dtype=str)
    # print(npFsmTable)

    return npFsmTable


def getGesture(stateName, fsmTable):
    gestures = []
    # print("========================================")
    # print("given state: ",stateName)
    for i in range(0, len(fsmTable)):
        if str(stateName) == str(fsmTable[i, 0]):
            gestures.append(fsmTable[i, 1])

    return np.array(gestures)


def getNextState(stateName, actionName, fsmTable):
    nextState = stateName
    print("========================================")
    print("given state: ", stateName)
    print("given action: ", actionName)
    for i in range(0, len(fsmTable)):
        if str(stateName) == str(fsmTable[i, 0]) and str(actionName) == str(fsmTable[i, 1]):
            # print("next state: ", fsmTable[i,2])
            nextState = fsmTable[i, 2]

    return nextState


def checkFSM(gesture, fsmTable):
    global currentState
    gesture = "-1"
    maxProb = -1
    idGest = -1

    currentState = getNextState(currentState, gesture, fsmTable)

    # print(currentState)

    return gesture

def checkhandIn2(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image):

    global handin
    handin = False

    if boxCounter == True:

        #dRefMaxF = deptRef - 0.15
        dRefMaxF = deptRef - 0.2
        # dRefMinF = dRefVal - 0.1

        boxXup = int(midx - (w * 3))
        boxYup = int(midy - (h * 1.5))

        boxXdn = int(midx + (w * 1))
        boxYdn = int(midy + (h * 2))

        if boxXup < 1: boxXup = 0
        if boxXdn >= 848: boxXdn = 848

        if boxYup < 1: boxYup = 0
        if boxYdn >= 480: boxYdn = 480

        for i in range(boxXup, boxXdn):
            for j in range(boxYup, boxYdn):
                if depth_imageS.item(j, i) > dRefMaxF or depth_imageS.item(j, i) == 0:
                    boxColor.itemset((j, i, 0), 0)
                    boxColor.itemset((j, i, 1), 0)
                    boxColor.itemset((j, i, 2), 0)

        roi_boxCounter = boxColor[boxYup.__int__():boxYdn.__int__(), boxXup.__int__():boxXdn.__int__()]

        graybox = cv2.cvtColor(roi_boxCounter, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(graybox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        idx = 0
        for cnt in contours:
            idx += 1
            xb, yb, wb, hb = cv2.boundingRect(cnt)
            areaCnt = wb * hb
            if areaCnt > 500:
                # print(areaCnt)
                handin = True
                break

        cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                      (0, 255, 255), 2)

    return handin


def checkhandIn(boxCounter, deptRef, xup, yup, xdn, ydn, depth_imageS, draw_image):

    global handin

    if boxCounter == True:

        #dRefMaxF = deptRef - 0.15
        dRefMaxF = deptRef - 0.15
        #dRefMinF = dRefVal - 0.1

        boxXup = int(xup + 50)
        boxYup = int(yup + 50)

        boxXdn = int(xdn - 50)
        boxYdn = int(ydn - 50)

        ctr = 0

        for i in range(boxXup, boxXdn):
            for j in range(boxYup, boxYdn):
                if depth_imageS.item(j, i) < dRefMaxF and depth_imageS.item(j, i) > 0.3  or depth_imageS.item(j, i) == 0:
                    ctr = ctr+1
                    #boxColor.itemset((j, i, 0), 0)
                    #boxColor.itemset((j, i, 1), 0)
                    #boxColor.itemset((j, i, 2), 0)

        #roi_boxCounter = boxColor[boxYup.__int__():boxYdn.__int__(), boxXup.__int__():boxXdn.__int__()]
        #graybox = cv2.cvtColor(roi_boxCounter, cv2.COLOR_BGR2GRAY)
        #_, contours, _ = cv2.findContours(graybox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #print("counter:",ctr)

        if ctr > 9000:
            handin = True
        else:
            handin = False

        cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                          (0, 0, 255), 2)

    return handin

def getFace(faces):
    print(faces)

def translateAvailableGest(availableGest, classGest):

    ctr = 0
    translateGest = []
    translateIGest = []


    for i in range(0,len(classGest)):
        for j in range(0, len(availableGest)):
            if str(availableGest[j]) == classGest[i]:
                translateGest.append(i)
                ctr = ctr+1

    for i in range(0, 10):
        ctr2 = 0
        for j in range(0,len(translateGest)):
            if str(translateGest[j]) == str(i):
                ctr2 = 1
        if ctr2 == 0:
            translateIGest.append(i)
            #print(i)

    return translateGest, translateIGest

def manipWeight(weightI, gAvailIgnore, mulitp):

    maxClass = 10
    # change one column to 0 and see the probability result
    for b in range(maxClass):
        for a in range(len(gAvailIgnore)):
            if gAvailIgnore[a] == b:
                # weightDense2[:, b] = np.multiply(weightDense2[:, b],100)
                weightI[b] = np.multiply(weightI[b], mulitp)
    return weightI

def main():
    global handin
    global currentState

    # img_rows, img_cols, maxFrames = 32, 32, 100
    img_rows, img_cols, maxFrames = 50, 50, 55
    depthFrame = 0
    cameraHeightR = 480
    cameraWidthR = 848
    # cameraWidthR = 640
    frameRateR = 60
    # frameRateR = 30

    # crop parameter
    xupam = 350
    yupam = 200

    xdpam = 250
    ydpam = 300

    #1,3,4,5,8,9,12,13,14,15
    classGest = ['1', '12', '13', '14', '15', '3', '4', '5', '8', '9']
    nameGest = ['call', 'scroll up', 'scroll down', 'right', 'left', "like", 'play/pause',
                'close', 'click', 'back']
    #classGest = ['11', '12', '13','4','8']
    #nameGest = ['back', 'scroll up', 'scroll down', 'close 4', 'click 8']
    delayGest = 5
    delayBol = False
    backgroundRemove = True
    boxCounter = True

    # load the model and weight
    json_file = open('3dcnnresult/3dcnnmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("3dcnnresult/3dcnnmodel.hd5")
    print("Loaded model from disk")

    loaded_model.compile(loss=categorical_crossentropy,
                         optimizer='rmsprop', metrics=['accuracy'])

    conf = loaded_model.model.get_config()


    shapeInput, ap = loaded_model.model.get_layer(name="dense_2").input_shape
    shapeOutput, sp = loaded_model.model.get_layer(name="dense_2").output_shape
    weightDense2 = loaded_model.model.get_layer(name="dense_2").get_weights()[0]
    weightDense22I = loaded_model.model.get_layer(name="dense_2").get_weights()[1]
    weightDense22A = loaded_model.model.get_layer(name="dense_2").get_weights()[1]

    # load the FSM
    ytfsmfile = "youtube_fsm.txt"
    ytFsmTable = loadFSM(ytfsmfile)



    updatedWeight = False
    updatedWeight2 = False


    ''''
    print("========================New weight I ==================================")
    print(weightDense22I)
    print("========================New weight A ==================================")
    print(weightDense22A)
    '''

    #print(newWeight[0])


    #NweightDense2 = loaded_model.model.get_layer(name="dense_2").get_weights()[0]


    #modelConfig = loaded_model.get_config()
    #print(modelConfig)

    # setup cv face detection
    face_cascade = cv2.CascadeClassifier('G:/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')

    # setup Realsense
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, cameraWidthR, cameraHeightR, rs.format.z16, frameRateR)
    config.enable_stream(rs.stream.color, cameraWidthR, cameraHeightR, rs.format.bgr8, frameRateR)

    # if using background removal

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print "Depth Scale is: " , depth_scale

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # for text purpose
    imgTxt = np.zeros((480, 400, 3), np.uint8)
    txt = "OpenCV"
    txtLoad = "["
    txtDelay = "["
    txtRecord = "Capture"
    txtDel = "Delay"
    txtProbability = "0%"
    font = cv2.FONT_HERSHEY_SIMPLEX

    framearray = []
    ctrDelay = 0
    channel = 1
    gestCatch = False
    gestStart = False

    startTime = 0
    endTime = 0

    x, y, w, h = 0, 0, 0, 0

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:

            dataImg = []

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if (backgroundRemove == True):
                # Remove background - Set pixels further than clipping_distance to grey
                # grey_color = 153
                grey_color = 0
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
                # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color,
                                      color_image)

                color_image = bg_removed
                draw_image = color_image

            else:
                draw_image = color_image

            # face detection here


            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            if gestCatch == False:
                faces = face_cascade.detectMultiScale(gray, 1.1, 2)
                #print("face: ",len(faces))

                ctr = 0
                idxFace = -1
                minDist = 9999

                if len(faces) > 0:
                    for f in faces:
                        xh, yh, wh, hh = f
                        farea = wh * hh

                        midxf = int(xh + (wh * 0.5))
                        midyf = int(yh + (hh * 0.5))

                        depth_imageS = depth_image * depth_scale
                        deptRef = depth_imageS.item(midyf, midxf)

                        if deptRef <= minDist:
                            idxFace = ctr
                            minDist = deptRef

                        ctr = ctr+1
                    #print("id face", idxFace)

                    if idxFace >= 0:
                        x, y, w, h = faces[idxFace]
                        cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            

            # crop the face then pass to resize

            midx = int(x + (w * 0.5))
            midy = int(y + (h * 0.5))

            xUp = (x - (w * 3))
            yUp = (y - (h * 1.5))

            xDn = ((x + w) + (w * 1))
            yDn = ((y + h) + (h * 2))

            if xUp < 1: xUp = 0
            if xDn >= cameraWidthR: xDn = cameraWidthR

            if yUp < 1: yUp = 0
            if yDn >= cameraHeightR: yDn = cameraHeightR

            if handin == False:
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 0, 255),
                              2)
            else:
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 255, 0),
                              2)
            cv2.circle(draw_image, (midx.__int__(), midy.__int__()), 10, (255, 0, 0))

            # region of interest
            roi_gray = gray[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]



            #print(cv2.useOptimized())
            '''''


            stateText = "State " + str(currentState)
            cv2.putText(imgTxt, stateText, (10, 200), font, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            avtext = "Available Gesture"
            cv2.putText(imgTxt, avtext, (10, 230), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            for i in range(0, len(availableGest)):
                availGestText = "G" + availableGest[i] + ", "
                cv2.putText(imgTxt, availGestText, (10 + (i * 80), 260), font, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

            '''

            # find the depth of middle point of face
            if backgroundRemove == True and gestCatch == False:
            #if backgroundRemove == True:
                e1 = cv2.getTickCount()

                depth_imageS = depth_image * depth_scale
                deptRef = depth_imageS.item(midy, midx)
                # print(clipping_distance)

                clipping_distance = (deptRef + 0.2) / depth_scale

                boxColor = color_image.copy()

                #handin = checkhandIn(boxCounter, deptRef, xUp, yUp, xDn, yDn, depth_imageS, draw_image)
                handin = checkhandIn2(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image)

                #print(handin)

                e2 = cv2.getTickCount()

                times = (e2 - e1) / cv2.getTickFrequency()
                #print(times)

                #handin = False


                if handin == True:
                    gestStart = True
                else:
                    gestStart = False

            # print("gest start = ", gestStart)

            if delayBol == False and gestStart == True:
                # your code execution

                if depthFrame < maxFrames:

                    frame = cv2.resize(roi_gray, (img_rows, img_cols))
                    framearray.append(frame)
                    depthFrame = depthFrame + 1
                    txtLoad = txtLoad + "["

                    gestCatch = True

                    # print(depthFrame)

                if depthFrame == maxFrames:
                    dataImg.append(framearray)
                    xx = np.array(dataImg).transpose((0, 2, 3, 1))
                    X = xx.reshape((xx.shape[0], img_rows, img_cols, maxFrames, channel))
                    X = X.astype('float32')
                    print('X_shape:{}'.format(X.shape))

                    #==================== Update the Weight =======================================
                    newWeightI = []
                    newWeightA = []

                    # find the available gesture in the current state
                    availableGest = getGesture(currentState, ytFsmTable)
                    print(availableGest)
                    availG, ignoreG = translateAvailableGest(availableGest, classGest)
                    print(availG)
                    print(ignoreG)


                    if updatedWeight:
                        weightI = manipWeight(weightDense22I, ignoreG, 1000)

                        newWeightI.append(weightDense2)
                        newWeightI.append(weightI)

                    if updatedWeight2:
                        maxClass = 10
                        weightA = manipWeight(weightDense22A, availG, 1000)

                        newWeightA.append(weightDense2)
                        newWeightA.append(weightA)

                    #=================================================================================

                    if updatedWeight == False and updatedWeight2 == False:
                        newWeightI.append(weightDense2)
                        newWeightI.append(weightDense22A)

                    loaded_model.model.get_layer(name="dense_2").set_weights(newWeightI)

                    # do prediction
                    resc = loaded_model.predict_classes(X)[0]
                    res = loaded_model.predict_proba(X)[0]

                    # find the result of prediction gesture
                    resultC = classGest[resc]
                    nameResultG = nameGest[resc]

                    for a in range(0, len(res)):
                        print("Gesture {}: {} ".format(str(nameGest[a]), str(res[a] * 100)))

                    print("===============================================================")

                    if updatedWeight2:
                        loaded_model.model.get_layer(name="dense_2").set_weights(newWeightA)

                        # do prediction
                        resc2 = loaded_model.predict_classes(X)[0]
                        res2 = loaded_model.predict_proba(X)[0]

                        # find the result of prediction gesture
                        resultC2 = classGest[resc2]
                        nameResultG2 = nameGest[resc2]

                        for a in range(0, len(res2)):
                            print("Gesture {}: {} ".format(str(nameGest[a]), str(res2[a] * 100)))


                    # show text of gesture result
                    imgTxt = np.zeros((480, 400, 3), np.uint8)
                    #txt = "Gesture-" + str(resultFsm)
                    if updatedWeight2:
                        if res2[resc2] > res[resc]:
                            txt = "ignored gesture"
                            act = -1
                        else:
                            txt = nameResultG
                            act = resultC
                    else:
                        txt = nameResultG
                        act = resultC

                    print(act)

                    # check with FSM for finding the next state
                    currentState = getNextState(currentState, act, ytFsmTable)
                    print(currentState)

                    framearray = []
                    #dataImg = []
                    txtLoad = ""
                    depthFrame = 0

                    gestCatch = False
                    handin = False
                    delayBol = True

                
                cv2.putText(imgTxt, txtLoad, (10, 20), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txtRecord, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txt, (10, 160), font, 2, (255, 255, 255), 2, cv2.LINE_AA)


            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # print(delayBol)
            if delayBol == True:
                ctrDelay = ctrDelay + 1
                txtDelay = txtDelay + "["
                txtDel = "Delay"
                cv2.putText(imgTxt, txtDelay, (10, 70), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txtDel, (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if ctrDelay == delayGest:
                    ctrDelay = 0
                    txtDelay = ""
                    delayBol = False

                    gestCatch = False
                    handin = False
                    gestStart = False

            draw_image = cv2.flip(draw_image,1)
            # Stack both images horizontally
            images = np.hstack((draw_image, imgTxt))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()
