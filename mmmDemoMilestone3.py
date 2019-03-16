#   __  __             _           _   __  __           _        __  __            _     _            
#  |  \/  | __ _  __ _(_) ___ __ _| | |  \/  |_   _ ___(_) ___  |  \/  | __ _  ___| |__ (_)_ __   ___ 
#  | |\/| |/ _` |/ _` | |/ __/ _` | | | |\/| | | | / __| |/ __| | |\/| |/ _` |/ __| '_ \| | '_ \ / _ \
#  | |  | | (_| | (_| | | (_| (_| | | | |  | | |_| \__ \ | (__  | |  | | (_| | (__| | | | | | | |  __/
#  |_|  |_|\__,_|\__, |_|\___\__,_|_| |_|  |_|\__,_|___/_|\___| |_|  |_|\__,_|\___|_| |_|_|_| |_|\___|
#                |___/                                                                                
# 
#
# Welcome to Magical Music Machine's implementation for the code of Milestone 3!
#
# The basic structure of the code is as follows:
#
#	1. Import necessary libraries and set up data structures used in the digital signal processing (DSP) and machine learning (ML)
#
#	2. Establish helper functions that will be used to assess DSP and ML flags
#
#	3. While loop that sends the frames from the webcam through OpenPose and sends the keypoint skeletal data to the appropriate functions
#
#	
#   _     _     _ _                                            _   ____        _          ____  _                   _                       
#  / |   | |   (_) |__  _ __ __ _ _ __ _   _    __ _ _ __   __| | |  _ \  __ _| |_ __ _  / ___|| |_ _ __ _   _  ___| |_ _   _ _ __ ___  ___ 
#  | |   | |   | | '_ \| '__/ _` | '__| | | |  / _` | '_ \ / _` | | | | |/ _` | __/ _` | \___ \| __| '__| | | |/ __| __| | | | '__/ _ \/ __|
#  | |_  | |___| | |_) | | | (_| | |  | |_| | | (_| | | | | (_| | | |_| | (_| | || (_| |  ___) | |_| |  | |_| | (__| |_| |_| | | |  __/\__ \
#  |_(_) |_____|_|_.__/|_|  \__,_|_|   \__, |  \__,_|_| |_|\__,_| |____/ \__,_|\__\__,_| |____/ \__|_|   \__,_|\___|\__|\__,_|_|  \___||___/
#                                      |___/                                                                                                
#
# It requires OpenCV installed for Python
import time
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import tensorflow as tf
from multiprocessing import Process,Queue,Pipe
import OSC
import scipy as sp
c = OSC.OSCClient()
c.connect(('10.19.51.244', 57120))   # connect to SuperCollider on surface Josh


# Here are the 13 different poses our model was able to distinguish
class_names = ['RR', 'guitar-knees', 'bass', 'cowbell', 'piano', 'guitar', 'goats','clap','drums','dab','stand','throatcut','bow']

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
# This sets up the camera and net resolution for the OpenPose model.
# These dimensions were chosen for speed, as when we tried to increase the dimensions it was not fast enough for realtime feedback (1-2 fps).
# With these dimensions we get 11-12 fps.
params = dict()
params["model_folder"] = "../../../models/"
params['net_resolution'] = '128x96'
params['camera_resolution'] = '640x480'
params['camera'] = '1'
params['render_pose'] = 0

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

#Get the camera set up with the dimensions we want
stream = cv2.VideoCapture(1)
if (not stream.isOpened()):  # check if succeeded to connect to the camera
   print("Cam open failed");
else:
    stream.set(3,640);
    stream.set(4,480);


# Beginning of data structures
loop = 0
windowSize = 6 # just for reference

# List of the coords of the different people.
# These fill up with 50 elements at a time (25 keypoints each with an x and y coordinate). 
list1 = []
list2 = []
list3 = []

# This array is used to find the average of the delta of their movement over the last second or so
motion = {1:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 2:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 3:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
drumHistory = {'left':[0,0],'right':[0,0]}
ready = [0,0,0]
state = {'mode':'begin','players':{'guitar':0,'bass':0,'drums':0}}

# List of osc messages that get sent to the server.
# The second element in the array is the number of seconds to wait before sending a similar message 
osc = {'bang':[time.time(),1],'highkick':[time.time(),1],'bigstrumguitar':[time.time(),.2],'bigstrumbass':[time.time(),2],'crowd':[time.time(),2],'strum':[time.time(),.2],'strums':[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],4],'drums1':[time.time(),.1],'drums2':[time.time(),.1],'drums3':[time.time(),.1],'drums4':[time.time(),.1],'drums5':[time.time(),.1],'drums6':[time.time(),.1],'dab':[time.time(),3],'piano':[time.time(),.1],'cowbell':[time.time(),.2],'goats':[time.time(),3],'bow':[time.time(),1],'dog':[time.time(),1.5]}
tempMovements = []

# Body keypoints as coordinates for easier reference
headX = 0 
headY = 1
neckX = 2
neckY = 3
rShoulderX = 4
rShoulderY = 5
rElbowX = 6
rElbowY = 7
rWristX = 8
rWristY = 9
lShoulderX = 10
lShoulderY = 11
lElbowX = 12
lElbowY = 13
lWristX = 14
lWristY = 15
pelvisX = 16
pelvisY = 17
rHipX = 18
rHipY = 19
rKneeX = 20
rKneeY = 21
rAnkleX = 22
rAnkleY = 23
lHipX = 24
lHipY = 25
lKneeX = 26
lKneeY = 27
lAnkleX = 28
lAnkleY = 29
frame6 = 250
frame5 = 200
frame4 = 150
frame3 = 100
frame2 = 50
frame1 = 0


# Starting OpenPose
opWrapper = op.WrapperPython()
print(params)
opWrapper.configure(params)
opWrapper.start()
oldtime = time.time()

#	  ____      _   _      _                   _____                 _   _                 
#	 |___ \    | | | | ___| |_ __   ___ _ __  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
#	   __) |   | |_| |/ _ \ | '_ \ / _ \ '__| | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#	  / __/ _  |  _  |  __/ | |_) |  __/ |    |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#	 |_____(_) |_| |_|\___|_| .__/ \___|_|    |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#	                        |_|                                                            

#Function to send data, requires a string that gets sent to Supercollider
def packageAndSend(feature):
	global osc
	if (osc[feature][1] != 0):
		if (time.time() - osc[feature][0] > osc[feature][1]):
			print('*************SENT TO MUSIC SERVER: ' + str(feature) + str(time.time() - osc[feature][0]))
			osc[feature][0] = time.time()
			oscmsg = OSC.OSCMessage()
			oscmsg.setAddress('/bang')
			oscmsg.append(feature)
			c.send(oscmsg)

#Returns the keypoints in a flat list normalized to a value between 0 and 1
def returnNormalizedKeypoints(keypoints):
    temp = []
    for i in range(len(keypoints)):
        if (keypoints[i,0] != 0):
            temp.append(keypoints[i,0]/640) # x value normalized (640 pixels wide)
        else:
            temp.append(-1)
        if (keypoints[i,1] != 0):
            temp.append(keypoints[i,1]/480) # y value normalized (480 pixels high)
        else:
            temp.append(-1)
    #returned the normalized keypoint
    return temp

# Returns the percentage that x is between a and b,
# Assumes a is lower value and b is higher value
# Use this to find the fret level (how high your hand is in relation to your torso)
def returnPercentage(x,a,b):
    if (x >=a):
        return 0
    if (x <=b):
        return 1
    else:
        return (x-a)/(b-a)

# Returns the angle angle (in degrees) for p0p1p2 corner
# Used to find a power strum 
#   Inputs:
#    p0,p1,p2 - points in the form of [x,y]
def get_angle(p0, p1=np.array([0,0]), p2=None):
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

# This is where we pedict the user's pose with the ML model we trained
# Depending on the mode (Begin, Play), striking a pose will have different effects
def checkForPose(history, person):
    history = np.reshape(history, (-1, 50))
    a = np.expand_dims(history, axis=0)
    b = model.predict(a)
    blist = []
    for i in b[0]:
		blist.append(i)
    #print ('person ' + str(person) +':' +  class_names[blist.index(max(b[0]))] + " confidence: " + str(max(b[0])))
    if (max(b[0]) > .80):
	predictedPose = class_names[blist.index(max(b[0]))]	
	# If they just started the program, let them choose between guitar, bass, or drums
	# Once they are locked in, don't let them switch
	if (state['mode'] == 'begin'):	
		if (predictedPose == 'guitar' and state['players']['guitar'] ==0 and state['players']['bass'] != person and state['players']['drums'] != person):
			state['players']['guitar'] = person
			print('player ' + str(person) + 'selected guitar')
		elif(predictedPose == 'bass' and state['players']['bass'] ==0 and state['players']['guitar'] != person and state['players']['drums'] != person):
			state['players']['bass'] = person
			print('player ' + str(person) + 'selected bass')
		elif(predictedPose == 'drums' and state['players']['drums'] ==0 and state['players']['bass'] != person and state['players']['guitar'] != person):
			state['players']['drums'] = person
			print('player ' + str(person) + 'selected drums')
		# If they bow and have selected an instrument, start up the play mode!
		elif(predictedPose == 'bow'):
			if (state['players']['guitar'] ==person or state['players']['bass']==person or state['players']['drums']==person): 
				ready[person-1] = 1
				print('player ' + str(person) + 'is ready')
				if (ready[0]==1 and ready[1]==1 and ready[2]==1):
					print('heading into play mode!')
					state['mode'] = 'play'
					print('in play mode')
				else:
					#print('waiting on other players...')
					print('entering test mode')
					state['mode']='test'
					#packageAndSend('bang')
	#If they are in play mode ('test'), send bang to Supercollider with appropriate message
	elif state['mode'] == 'test':
		if (predictedPose == 'dab' and max(b[0]) > .95): #dab is sent too often, raise confidence requirement
			print('dab')
			packageAndSend('dab')
		elif (predictedPose == 'piano'):
			print('piano')
			packageAndSend('piano')
		elif (predictedPose == 'cowbell'):
			print('cowbell')			
			packageAndSend('cowbell')
		elif (predictedPose == 'goats'):
			print('goats')
			packageAndSend('goats')
		elif (predictedPose == 'guitar-knees'):
			print('dog')
			packageAndSend('dog')
		elif(predictedPose == 'throatcut'):
			print('throat' + str(max(b[0])))
			#if (max(b[0]) > .95):
				#packageAndSend('bow')			
				#os._exit(0) # Bow to exit the program  

# This function is for the person in drum mode, we check to see if they are drumming.
# We draw some lines around their torso and return the quadrant they are in (returns 0 if inside the box).
def checkDrumsDSP(side,handX,handY,history,currentFrame):
	yMax = max(history[currentFrame + lShoulderY],history[currentFrame + rShoulderY])
	yMin = min(history[currentFrame + lHipY],history[currentFrame + rHipY])
	xMax = max(history[currentFrame + lShoulderX],history[currentFrame + lHipX])
	xMin = min(history[currentFrame + rShoulderX],history[currentFrame + rHipX])
	#adjust box after testing it out to avoid less false positives
	yMax = yMax*.9
	yMin = yMin*.9
	xMax = xMax*1.1
	xMin = xMin*.9
	#Here we check to see if their wrist is out of the safe zone(torso). 
	#If it is out of the box, we send a drum hit depending on the location.
	if (history[currentFrame + handY] < yMax or history[currentFrame + handY] > yMin or history[currentFrame + handX] < xMin or history[currentFrame + handX] > xMax):
		if (history[currentFrame + handY] < yMax):
			if (history[currentFrame + handX] < xMin):
			    #print(str(side) + ' quadrant 1')
			    return 1
			elif(history[currentFrame + handX] > xMax):
			    #print(str(side) + 'quadrant 3')
			    return 3
			else:
			    #print(str(side) + 'quadrant 2')
			    return 2
		else: #left hand is below waist
			if (history[currentFrame + handX] < xMin):
			    #print(str(side) + 'quadrant 4')
			    return 4
			elif(history[currentFrame + handX] > xMax):
			    #print(str(side) + 'quadrant 6')
			    return 6
			else:
			   # print(str(side) + 'quadrant 5')
			    return 5
		return 0

# This function is for the guitar and bass players.
# It draws a line between their hip and shoulder,
# then checks to see if their wrist has crossed over that line between frames 
def checkStrums(history,person,currentFrame, previousFrame):
	if (person != state['players']['drums']):
		strums = []
		wrist1 = [0,0]
		wrist2 = [0,0]
		word = 'lol'
		leftOrRight =0;
		fretHand = [0,0]
		hip = [0,0]
		shoulder = [0,0]
		if (person == state['players']['bass']):
			wrist1 = [lWristX,lWristY]
			wrist2 = [rShoulderX,rShoulderY]
			hip = [lHipX,lHipY]
			shoulder = [rShoulderX,rShoulderY]
			word = 'bass'
			fretHand = [rWristX,rWristY]
			leftOrRight = .99
		elif (person == state['players']['guitar']):
			wrist1 = [rWristX,rWristY]
			wrist2 = [lShoulderX,lShoulderY]
			hip = [rHipX,rHipY]
			shoulder = [lShoulderX,lShoulderY]
			word = 'guitar'
			fretHand = [lWristX,lWristY]
			leftOrRight = 0
		for i in (currentFrame,previousFrame):
			x = history[i  + wrist1[0]]
			y = history[i  + wrist1[1]]
			x1 = history[i + wrist2[0]]
			y1 = history[i + wrist2[1]]
			x2 = history[i + hip[0]]
			y2 = history[i + hip[0]]
			#Here is where we see which side of the line its on
			strums.append((x-x1)*(y2-y1)-(y-y1)*(x2-x1))
		#If the wrist was on one side of the line and then on the other in the next frame, send a strum message.
		if (strums[0] > 0 and strums[1] < 0) or (strums[0] < 0 and strums[1] > 0):
			if word=='bass':
				packageAndSend('bigstrumbass')
			else: 
				packageAndSend('strum')
        
		currentAngle = get_angle([history[currentFrame+wrist1[0]],history[currentFrame+wrist1[1]]],[history[currentFrame+shoulder[0]],history[currentFrame+shoulder[1]]],[leftOrRight,history[currentFrame+shoulder[1]]])
		previousAngle = get_angle([history[previousFrame+wrist1[0]],history[previousFrame+wrist1[1]]],[history[previousFrame+shoulder[0]],history[previousFrame+shoulder[1]]],[leftOrRight,history[previousFrame+shoulder[1]]])
		previousFrame2 = previousFrame-50
		previousAngle2 = get_angle([history[previousFrame2+wrist1[0]],history[previousFrame2+wrist1[1]]],[history[previousFrame2+shoulder[0]],history[previousFrame2+shoulder[1]]],[leftOrRight,history[previousFrame2+shoulder[1]]])
		#print(currentAngle)
		if (currentAngle>=90 and currentAngle <=170) and (previousAngle<=-90 or previousAngle2 <=-90):
		    #print ('powerStrum ' + word)
		    packageAndSend('bigstrum' + word)
	
	#Check Frets here too, because we already have the person identified
	# Didn't use this in the demo because of the computational load of doing it to the music
	#print("fret level " + word + ": " + str(returnPercentage(history[currentFrame+fretHand[1]],history[currentFrame+pelvisY],history[currentFrame+headY])))	

# Check for the major DSP actions
# Each person sends their keypoint data to this function each frame
def checkForDSP(history,person,currentFrame,previousFrame):
    #highkick
    #If their ankle goes above the opposite knee, send highkick message
    if (history[currentFrame + lAnkleY] < history[currentFrame + rKneeY] or history[currentFrame + rAnkleY] < history[currentFrame + lKneeY]):
        packageAndSend('highkick')
		print('person ' + str(person) + 'highkick')

    #find the jump 
    percentageAnkleGain = returnPercentage(history[currentFrame+ lAnkleY],history[previousFrame+lAnkleY],history[previousFrame+headY])
    percentageNeckGain = returnPercentage(history[currentFrame + neckY],history[previousFrame+neckY],history[previousFrame+headY])
    percentagePelvisGain = returnPercentage(history[currentFrame + pelvisY],history[previousFrame+pelvisY],history[previousFrame+headY])
    if ( percentageAnkleGain > .004 and percentageNeckGain > .1 and percentagePelvisGain >.1):
        packageAndSend('crowd')

     # Explored but ultimately not used.
     # A way to gesturally control the interface.
        #if (list1[214] < list1[216] and (abs(list1[208]-list1[204])>abs(list1[214]-list1[202]))):
        #   print('reaching right')
        #if (list1[208] > list1[216] and (abs(list1[214]-list1[210])>abs(list1[208]-list1[202]))):  
            
        #   print('reaching left')

    # Update randomness of motion if we got all the major keypoints
    # Not used in demo as we didn't have access to the musical latent space that controls note density.
    #if not (-1 in history[previousFrame:previousFrame+lAnkleX] or -1 in history[currentFrame:currentFrame+lAnkleX]):
     #   check = np.sum(abs(np.subtract(history[previousFrame:previousFrame+lAnkleX],history[currentFrame:currentFrame+lAnkleX])))
      #  motion[person].pop(0)
       # motion[person].append(check)
	#print("motion: " + str(np.mean(motion[person])))


	# If they are the drummer, add their keypoints to the apprpriate data structure.
	# Also check to see if they are drumming
    if (person == state['players']['drums']):#drummer
		if len(drumHistory['left']) > 5:
			drumHistory['left'].pop(0)
		if len(drumHistory['right']) > 5:
			drumHistory['right'].pop(0)
		drumHistory['left'].append(checkDrumsDSP('left',lWristX,lWristY,history,currentFrame))
		drumHistory['right'].append(checkDrumsDSP('right',rWristX,rWristY,history,currentFrame))
		# If their current frame is in a quadrant and the previous one was in the 'No Drum Zone'
		# More potential innovation here.
		if drumHistory['left'][-1] and not drumHistory['left'][-2]:
				print(str(person) + 'left hand ' + str(drumHistory['left'][-1]))
				packageAndSend('drums' + str(drumHistory['left'][-1]))
	        if drumHistory['right'][-1] and not drumHistory['right'][-2]:
				print(str(person) + 'right hand ' + str(drumHistory['right'][-1]))
				packageAndSend('drums' + str(drumHistory['right'][-1]))
        


 #  _____    ____             _        __        ___     _ _        _                      
 # |___ /   | __ )  ___  __ _(_)_ __   \ \      / / |__ (_) | ___  | |    ___   ___  _ __  
 #   |_ \   |  _ \ / _ \/ _` | | '_ \   \ \ /\ / /| '_ \| | |/ _ \ | |   / _ \ / _ \| '_ \ 
 #  ___) |  | |_) |  __/ (_| | | | | |   \ V  V / | | | | | |  __/ | |__| (_) | (_) | |_) |
 # |____(_) |____/ \___|\__, |_|_| |_|    \_/\_/  |_| |_|_|_|\___| |_____\___/ \___/| .__/ 
 #                      |___/                                                       |_|    

datum = op.Datum()
while True:
    # Need the loading of our model to happen a bit after the program loads 
    # Otherwise it throws a memory error.
    # Much frustration to figure this out.
    if loop ==3:
        model=tf.keras.models.load_model('model_size6_aug_2.h5')
    ret,img = stream.read()

   	#If we can an image, load emblaceAndPop in OpenPose
    if (img.any()):
        #print(img)
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])

    keypoints = datum.poseKeypoints # chop off the confidence levels
    #print(keypoints.shape)

    #If we see someone, add the keypoints to their history
    if (keypoints.shape !=()):  
        list1 = list1 + returnNormalizedKeypoints(keypoints[0])
        #two people     
        if (keypoints.shape[0]>1):
            list2 = list2 + returnNormalizedKeypoints(keypoints[1])
        #three people
        if (keypoints.shape[0]>2):
            list3 = list3 + returnNormalizedKeypoints(keypoints[2])

        #checking to see if we have enough data on person 1 to make a prediction
        # We check DSP every frame but ML is every other frame to save compute time and get more fps.
        if len(list1)==250:
		    if (state['mode']=='test'):
		        checkStrums(list1,1,frame5,frame4)
			checkForDSP(list1,1,frame5,frame4)
        elif len(list1) ==300: #300 is because 6 seconds of data, 25 (x,y) coords per second...aka  windowSize*50
            if state['mode'] == 'begin':
				checkForPose(list1,1)
		    else:
				checkForPose(list1,1)
		    if (state['mode']=='test'):
				checkStrums(list1,1,frame5,frame4)
				checkForDSP(list1,1,frame6,frame5)
            list1 = list1[100:] #delete the last 2 frames

        #checking to see if we have enough data on person 2 to make a prediction
        if len(list2)==250:
		    if (state['mode']=='test'):
		        checkStrums(list2,2,frame5,frame4)
	        elif len(list2) ==300:
	            checkForPose(list2,2)
		    if (state['mode']=='test'):
				checkStrums(list2,2,frame5,frame4)
				checkForDSP(list2,2,frame6,frame5)
	            list2 = list2[100:] #delete the last 2 frames

	    #checking to see if we have enough data on person 3 to make a prediction
        if len(list3)==250:
            if (state['mode']=='test'):
				checkForDSP(list3,3,frame5,frame4)
        elif len(list3) ==300:
            checkForPose(list3,3)
		    if (state['mode']=='test'):
				checkStrums(list3,3,frame5,frame4)
				checkForDSP(list3,3,frame6,frame5)
	            list3 = list3[100:] #delete the last 2 frames
    loop+=1
    #if loop == 100:
    #   break
