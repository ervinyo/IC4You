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


def checkFSM(avGesture, probAvGesture, resGesture, probResGesture, fsmTable):
    global currentState
    gesture = "-1"
    maxProb = -1
    idGest = -1

    # if the result predicted gesture is one of the available gesture in this state and if it have probability higher
    # than 80%?? then it will return the result predicted gesture, then update the current state by finding the next
    # state

    # if the predicted gesture is not one of the available gesture, return the gesture that have the highest probability
    # from the available gesture, update the current state

    # if the predicted gesture is one of the available gesture, but not in the proba higher than 80%??

    # or maybe simply return the highest probability from the available gesture then update the current state (easy one)

    for i in range(0, len(probAvGesture)):
        if probAvGesture[i] > maxProb:
            maxProb = probAvGesture[i]
            idGest = i

    if idGest > -1:
        gesture = avGesture[idGest]
        currentState = getNextState(currentState, gesture, fsmTable)

    # print(currentState)

    return gesture


def checkhandIn(boxCounter, deptRef, xup, yup, xdn, ydn, depth_imageS, draw_image):

    global handin

    if boxCounter == True:

        #dRefMaxF = deptRef - 0.15
        dRefMaxF = deptRef - 0.2
        #dRefMinF = dRefVal - 0.1

        boxXup = int(xup + 50)
        boxYup = int(yup + 50)

        boxXdn = int(xdn - 50)
        boxYdn = int(ydn - 50)

        ctr = 0

        for i in range(boxXup, boxXdn):
            for j in range(boxYup, boxYdn):
                if depth_imageS.item(j, i) < dRefMaxF or depth_imageS.item(j, i) == 0:
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

        cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                          (0, 0, 255), 2)

    return handin


def main():
    global handin

    # img_rows, img_cols, maxFrames = 32, 32, 100
    img_rows, img_cols, maxFrames = 32, 32, 55
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

    classGest = ['1', '11', '12', '13', '4', '5', '7', '8']
    nameGest = ['call', 'back', 'scroll up', 'scroll down', 'close 4', 'close 5', 'click 7', 'click 8']
    delayGest = 10
    delayBol = False
    backgroundRemove = True
    boxCounter = True

    # load the FSM
    fbfsmfile = "facebook_fsm.txt"
    fbFsmTable = loadFSM(fbfsmfile)

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
                depth_image_3d = np.dstack(
                    (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
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

                if len(faces) > 0:
                    for f in faces:
                        xh, yh, wh, hh = f
                        farea = wh * hh
                        if farea > 9000 and farea < 18000:
                            x, y, w, h = f
            

            # crop the face then pass to resize
            
            cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            
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

            # find the available gesture in the current state
            availableGest = getGesture(currentState, fbFsmTable)
            # print(availableGest[0])

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
                e1 = cv2.getTickCount()

                depth_imageS = depth_image * depth_scale
                deptRef = depth_imageS.item(midy, midx)
                # print(clipping_distance)

                clipping_distance = (deptRef + 0.2) / depth_scale

                handin = checkhandIn(boxCounter, deptRef, xUp, yUp, xDn, yDn, depth_imageS, draw_image)

                e2 = cv2.getTickCount()

                times = (e2 - e1) / cv2.getTickFrequency()
                #print(times)


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

                if depthFrame == 1:
                    startTime = time.time()

                if depthFrame == maxFrames:
                    dataImg.append(framearray)
                    xx = np.array(dataImg).transpose((0, 2, 3, 1))
                    X = xx.reshape((xx.shape[0], img_rows, img_cols, maxFrames, channel))
                    X = X.astype('float32')
                    print('X_shape:{}'.format(X.shape))

                    endTime = time.time()

                    seconds = endTime - startTime
                    print("Time taken : {0} seconds".format(seconds))

                    fps = 55 / seconds;
                    print("Estimated frames per second : {0}".format(fps))



                    # do prediction
                    resc = loaded_model.predict_classes(X)[0]
                    res = loaded_model.predict_proba(X)[0]

                    
                    #find the probability of available gesture
                    #probaAvGest = []
                    #for b in range(0,len(availableGest)):
                        #for c in range(0,len(classGest)):
                            #if str(classGest[c]) == str(availableGest[b]):
                                #print("Gesture Available=%s, Probability=%s" % (str(availableGest[b]),res[c] * 100))
                               #probaAvGest.append(res[c]*100)

                    

                    #find the result of prediction gesture
                    resultC = classGest[resc]
                    nameResultG = nameGest[resc]
                    #find the proba of prediction gesture
                    #probaPrGest = res[resc] * 100
                    #print("Gesture Predicted=%s, Probability=%s" % (resultC, res[resc] * 100))

                    # check with FSM for the prediction result
                    # resultFsm = checkFSM(availableGest, np.array(probaAvGest), resultC, probaPrGest, fbFsmTable)

                    #resultFsm = resultC
                    resultFsm = 1

                    if resultFsm != "-1":

                        # show text of gesture result
                        imgTxt = np.zeros((480, 400, 3), np.uint8)
                        # txt = "Gesture-" + str(resultFsm)
                        txt = nameResultG;
                        # txtProbability = str(res[resc]*100)+"%"

                        
                        #for i in range(0, len(probaAvGest)):
                            #txtProbability = "Gesture" + availableGest[i] + ": " + str(round(probaAvGest[i], 4))
                            #cv2.putText(imgTxt, txtProbability, (10, 330 + (i * 30)), font, 1, (255, 255, 255), 2,
                                        #cv2.LINE_AA)
                        

                        framearray = []
                        # dataImg = []
                        txtLoad = ""
                        depthFrame = 0

                        gestCatch = False
                        handin = False
                        delayBol = True
                        # gestStart = False

                    else:
                        # show text of gesture result
                        imgTxt = np.zeros((480, 400, 3), np.uint8)
                        txt = "Not available gesture"
                        # txtProbability = str(res[resc] * 100) + "%"

                        framearray = []
                        # dataImg = []
                        txtLoad = ""
                        depthFrame = 0

                        gestCatch = False
                        handin = False
                        delayBol = True
                        # gestStart = False
                
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
