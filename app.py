from flask import Flask, jsonify, render_template, request
import traceback
from flask_restful import Resource, Api
import speech_recognition as sr
import webbrowser as wb
import time
import os

import matplotlib
matplotlib.use('AGG')

from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from keras import backend as K
#import videoto3d

import numpy as np
import pyrealsense2 as rs
import cv2


app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/youtube/')
def youtube():
    return render_template('youtube.html')	

@app.route('/listvideo/')
def listvideo():
    return render_template('listvideo.html')	


@app.route('/ajax1', methods=['GET', 'POST'])
def ajax1():
    try:
        user =  request.form.get('username')
        return "1"
    except Exception:
        return "error"
		
@app.route('/openapp', methods=['GET', 'POST'])	
def openapp():
    try:
        f = open ('result.txt','r')
        #print (f.read())
        return f.read()
    except Exception:
        return "error"

@app.route('/voice', methods=['GET', 'POST'])	
def voice():
    r = sr.Recognizer() 
    with sr.Microphone() as source:
        f = open ('result.txt','w')
        f.write("wait")
        #print ("Please wait a moment...  Calibrating microphone NOW~")
        # listen for 1 seconds and create the ambient noise energy level 
        r.adjust_for_ambient_noise (source, duration=2) 
        #print ("Now, please say something !!!")
        res = 'tes'    
        audio = r.listen (source)
    try:
        #f = open ('result.txt','w')
        text = r.recognize_google(audio, language="EN")
        if text == 'close':
            return "close"
        res = (text)
        
        #print (r.recognize_google(audio, language="EN"), file = f)
        #f_text =  text + ".com"
        #wb.get (chrome_path) .open (f_text)
        #wb.open('http://127.0.0.1:5000/' + f_text)
        #print ("What you said has been saved as [result.txt] :)")
        return res
    except sr.UnknownValueError:
        return ("0")
    except sr.RequestError as e:
        return ("0")
    #f.close()
#######################################################
########              Gesture              ############
#######################################################
#######################################################
#Global Variable



# setup Realsense
# Configure depth and color streams

# for text purpose
txt = "OpenCV"
txtLoad = "["
txtDelay = "["
txtRecord = "Capture"
txtDel = "Delay"
txtProbability = "0%"
font = cv2.FONT_HERSHEY_SIMPLEX


#######################################################
@app.route('/gesture', methods=['GET', 'POST'])
def gesture():
    #return('gesture')
    #global testDir
    #global img_rows, img_cols, maxFrames
    #global depthFrame
#crop parameter
    #global xupam
    #global yupam
    #global xdpam
    #global ydpam
    #global classGest
    #global delayGest
    #global delayBol
#load the model and weight
    #global json_file
    #global loaded_model_json
    #global loaded_model
#setup cv face detection
    #global face_cascade
# setup Realsense
# Configure depth and color streams
# for text purpose
    global txt
    global txtLoad
    global txtDelay
    global txtRecord
    global txtDel
    global txtProbability
    global font
    #global framearray
    #global ctrDelay
    #global channel
    #global gestCatch
    #global gestStart
    #global x,y,w,h
    #global vc
    #global rval , firstFrame
    #global heightc, widthc, depthcol
    #global imgTxt
    #global resultC
    #global count
    #global stat
    testDir = "videoTest/"
    img_rows, img_cols, maxFrames = 32, 32, 100
    depthFrame = 0
#crop parameter
    xupam = 350
    yupam = 200

    xdpam = 250
    ydpam = 300

    classGest = ['1','11','12','13','4','5','7','8']
    delayGest = 20
    delayBol = False
    framearray = []
    ctrDelay = 0
    channel = 1
    gestCatch = False
    gestStart = False

    x,y,w,h = 0,0,0,0
    count=0

#load the model and weight
    json_file = open('3dcnnresult/3dcnnmodel.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
	# load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("3dcnnresult/3dcnnmodel.hd5")
#setup cv face detection
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    vc = cv2.VideoCapture(0)
    rval , firstFrame = vc.read()
    heightc, widthc, depthcol = firstFrame.shape
    imgTxt = np.zeros((heightc, 400, 3), np.uint8)
    #print('tryyyyy1')
    stat =0

#print("Loaded model from disk")

    loaded_model.compile(loss=categorical_crossentropy,
                     optimizer='rmsprop', metrics=['accuracy'])
    while True:
        dataImg = []
        #vc = cv2.VideoCapture(0)
        #if stat ==0:
        #    vc = cv2.VideoCapture(0)
        rval, color_image = vc.read()
        #print(color_image)
        
        if color_image is None or np.all(color_image==0):
            
            stat = 0
        else:
            #print(depthFrame)
            stat = 1   
            #return('tryyyyy3')
            
            #cv2.imshow('RealSense', color_image)
            #cv2.waitKey(1)
            draw_image = color_image
            #return color_image
            #print(color_image)
            #face detection here
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 2)

            if len(faces) > 0 and gestCatch == False:
                
                gestStart = True
                x, y, w, h = faces[0]
            else:
                x, y, w, h = x, y, w, h
                stat = 0
            fArea = w*h
            #print(fArea)

            #if fArea > 3000:

            # crop the face then pass to resize
            #cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            midx = int(x + (w * 0.5))
            midy = int(y + (h * 0.5))

            xUp = (x - (w * 3))
            yUp = (y - (h * 1.5))

            xDn = ((x + w) + (w * 1))
            yDn = ((y + h) + (h * 2))

            if xUp < 1: xUp = 0
            if xDn >= widthc: xDn = widthc

            if yUp < 1: yUp = 0
            if yDn >= heightc: yDn = heightc

            #cv2.rectangle(draw_image, (xUp.__int__(), yUp.__int__()), (xDn.__int__(), yDn.__int__()), (0, 0, 255), 2)

            #cv2.circle(draw_image, (midx.__int__(), midy.__int__()), 10, (255, 0, 0))
            roi_color = color_image[yUp.__int__():yDn.__int__(), xUp.__int__():xDn.__int__()]


            if delayBol == False and gestStart == True:

                if depthFrame < maxFrames:
                    frame = cv2.resize(roi_color, (img_rows, img_cols))
                    framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    depthFrame = depthFrame+1
                    txtLoad = txtLoad+"["
                    count=count+1
                    gestCatch = True

                    #print(depthFrame)


                if depthFrame == maxFrames:
                    dataImg.append(framearray)
                    xx = np.array(dataImg).transpose((0, 2, 3, 1))
                    X = xx.reshape((xx.shape[0], img_rows, img_cols, maxFrames, channel))
                    X = X.astype('float32')
                    #print('X_shape:{}'.format(X.shape))


                    #do prediction
                    resc = loaded_model.predict_classes(X)[0]
                    res = loaded_model.predict_proba(X)[0]

                    resultC = classGest[resc]
                    #print("X=%s, Probability=%s" % (resultC, res[resc]*100))

                    #for r in range(0,8):
                    #    print("prob: " + str(res[r]*100))

                    #show text
                    #imgTxt = np.zeros((480, 400, 3), np.uint8)
                    #txt = "Gesture-" + str(resultC)
                    #txtProbability = str(res[resc]*100)+"%"

                    framearray = []
                    #dataImg = []
                    txtLoad = ""
                    depthFrame = 0

                    gestCatch = False
                    delayBol = True
                    
                #cv2.putText(imgTxt, txtLoad, (10, 20), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgTxt, txtRecord, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgTxt, txt, (10, 200), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgTxt, txtProbability, (10, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            #print(delayBol)
            if delayBol == True:
                ctrDelay = ctrDelay+1
                txtDelay = txtDelay + "["
                #txtDel = "Delay"
                #cv2.putText(imgTxt, txtDelay, (10, 70), font, 0.1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgTxt, txtDel, (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if ctrDelay == delayGest:
                    ctrDelay = 0
                    txtDelay = ""
                    delayBol = False

            # Stack both images horizontally
            #images = np.hstack((draw_image, imgTxt))

            # put the text here

            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', images)
            #cv2.waitKey(1)
            
            if count==maxFrames:
                vc.release()
                K.clear_session()
                return resultC
        
    #finally:
        #return('endtryyyyy')
        #return resultC
        #Stop streaming
        #pipeline.stop()
        #vc.release()
    vc.release()		
    
#######################################################


if __name__ == '__main__':
    #gesture()
    #vc = cv2.VideoCapture(0)
    app.run(debug = True)

