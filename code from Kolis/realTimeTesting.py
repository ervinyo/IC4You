import os

import matplotlib
matplotlib.use('AGG')

from keras.models import model_from_json
from keras.losses import categorical_crossentropy
import videoto3d

import numpy as np
import pyrealsense2 as rs
import cv2

handin = False


def checkhandIn(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image):

    global handin

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
    '''''
        if handin==True:
            cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                      (0, 255, 255), 2)
        else:
            cv2.rectangle(draw_image, (boxXup.__int__(), boxYup.__int__()), (boxXdn.__int__(), boxYdn.__int__()),
                          (0, 0, 255), 2)
    '''''
    return handin

def main():

    global handin

    #img_rows, img_cols, maxFrames = 32, 32, 100
    img_rows, img_cols, maxFrames = 32, 32, 55
    depthFrame = 0
    cameraHeightR = 480
    cameraWidthR = 848
    #cameraWidthR = 640
    frameRateR = 60
    #frameRateR = 30

    #crop parameter
    xupam = 350
    yupam = 200

    xdpam = 250
    ydpam = 300

    classGest = ['1','11','12','13','4','5','7','8']
    delayGest = 10
    delayBol = False
    backgroundRemove = True
    boxCounter = True



    #load the model and weight
    json_file = open('3dcnnresult/3dcnnmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("3dcnnresult/3dcnnmodel.hd5")
    print("Loaded model from disk")

    loaded_model.compile(loss=categorical_crossentropy,
                         optimizer='rmsprop', metrics=['accuracy'])

    #setup cv face detection
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
    clipping_distance_in_meters = 2 # 1 meter
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


    x,y,w,h = 0,0,0,0

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
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())


            if(backgroundRemove == True):
                # Remove background - Set pixels further than clipping_distance to grey
                #grey_color = 153
                grey_color = 0
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

                color_image = bg_removed
                draw_image = color_image


            else:
                draw_image = color_image


            #face detection here
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 2)

            if len(faces) > 0:
                if gestCatch == False and delayBol == False:

                    for f in faces:
                        xf, yf, wf, hf = f
                        farea = wf*hf
                        #print(farea)
                        if farea > 9000:
                            x, y, w, h = f
                            #gestStart = True
                            #x, y, w, h = faces[0]
            else:
                x, y, w, h = x, y, w, h

            fArea = w*h
            #print(fArea)

            #if fArea > 3000:

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
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 0, 255), 2)
            else:
                cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 255, 0),
                              2)
            cv2.circle(draw_image, (midx.__int__(), midy.__int__()), 10, (255, 0, 0))

            roi_color = color_image[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]



            #find the depth of middle point of face
            if backgroundRemove == True and gestCatch == False:

                depth_imageS = depth_image*depth_scale
                deptRef = depth_imageS.item(midy, midx)
                #print(clipping_distance)

                clipping_distance = (deptRef + 0.2)/depth_scale

                boxColor = color_image.copy()

                handin = checkhandIn(boxCounter, deptRef, midx, midy, w, h, depth_imageS, boxColor, draw_image)

                if handin == True:
                    gestStart = True
                else:
                    gestStart = False

            #print("gest start = ", gestStart)

            if delayBol == False and gestStart == True:

                if depthFrame < maxFrames:
                    frame = cv2.resize(roi_color, (img_rows, img_cols))
                    framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    depthFrame = depthFrame+1
                    txtLoad = txtLoad+"["

                    gestCatch = True

                    #print(depthFrame)


                if depthFrame == maxFrames:
                    dataImg.append(framearray)
                    xx = np.array(dataImg).transpose((0, 2, 3, 1))
                    X = xx.reshape((xx.shape[0], img_rows, img_cols, maxFrames, channel))
                    X = X.astype('float32')
                    print('X_shape:{}'.format(X.shape))


                    #do prediction
                    resc = loaded_model.predict_classes(X)[0]
                    res = loaded_model.predict_proba(X)[0]

                    resultC = classGest[resc]
                    print("X=%s, Probability=%s" % (resultC, res[resc]*100))


                    for r in range(0,8):
                        print("prob: " + str(res[r]*100))


                    #show text
                    imgTxt = np.zeros((480, 400, 3), np.uint8)
                    txt = "Gesture-" + str(resultC)
                    txtProbability = str(res[resc]*100)+"%"

                    framearray = []
                    #dataImg = []
                    txtLoad = ""
                    depthFrame = 0

                    gestCatch = False
                    handin = False
                    delayBol = True
                    #gestStart = False

                cv2.putText(imgTxt, txtLoad, (10, 20), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txtRecord, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txt, (10, 200), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgTxt, txtProbability, (10, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            #print(delayBol)
            if delayBol == True:
                ctrDelay = ctrDelay+1
                txtDelay = txtDelay + "["
                #txtDel = "Delay"
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
