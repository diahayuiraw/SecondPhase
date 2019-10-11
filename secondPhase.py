import numpy as np
import face_recognition
import cv2
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time

outputFrame = None
lock = threading.Lock()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade= cv2.CascadeClassifier('haarcascade_smile.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_nose.xml')

shafiq = face_recognition.load_image_file("shafiq.jpg")
shafiq_encode = face_recognition.face_encodings(shafiq)[0]

hafizal = face_recognition.load_image_file("hafizal.jpg")
hafizal_encode = face_recognition.face_encodings(hafizal)[0]

dalila = face_recognition.load_image_file("dalila.jpg")
dalila_encode = face_recognition.face_encodings(dalila)[0]

ayu = face_recognition.load_image_file("ayu.jpg")
ayu_encode = face_recognition.face_encodings(ayu)[0]

izzat = face_recognition.load_image_file("izzat.jpg")
izzat_encode = face_recognition.face_encodings(izzat)[0]

tinggy = face_recognition.load_image_file("atikah.jpg")
tinggy_encode = face_recognition.face_encodings(tinggy)[0]


known_faces = [
    shafiq_encode,
    hafizal_encode,
    dalila_encode,
    ayu_encode,
    izzat_encode,
    tinggy_encode
]


cap = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []


app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	frame_number = 0
	# grab global references to the video stream, output frame, and
	# lock variables
	global cap, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		# frame = vs.read()
		# frame = imutils.resize(frame, width=400)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# gray = cv2.GaussianBlur(gray, (7, 7), 0)
		
		# ---------------------------------------------------

		# ret, img = cap.read()
		# small_frame = cv2.resize(img, (0, 0), fx=0.2, fy=0.2) # 1/5
	    
		# rgb_frame = small_frame[:, :, ::-1]

		

		# face_locations = face_recognition.face_locations(rgb_frame)
		# face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
		# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# for face_encodings in face_encodings:
		# 	match = face_recognition.compare_faces(known_faces, face_encodings, tolerance=0.90)
		# 	name = None
	        
		# 	if match[0]:
		# 		name = "Shafiq"
		# 		cv2.imwrite( str(name) + '_muka.jpg', img)

		# 	elif match[1]:
		# 		name = "Hafizal"
		# 		cv2.imwrite( str(name) + '_muka.jpg', img)

		# 	elif match[2]:
		# 		name = "Dalila"
		# 		cv2.imwrite( str(name) + '_muka.jpg', img)

		# 	elif match[3]:
		# 		name = "Ayu"
		# 		cv2.imwrite( str(name) + '_muka.jpg', img)

		# 	elif match[4]:
		# 		name = "Izzat"
		# 		cv2.imwrite( str(name) + '_muka.jpg', img)

		# 	elif match[5]:
		# 		name = "Tinggy"
		# 		cv2.imwrite( str(name) + '_muka.jpg', img)

		# 	else:
		# 		name = "Unknown"
		# 		cv2.imwrite( str(name) + '_muka.jpg', img)
			
		# 	face_names.append(name)

		# for (top, right, bottom, left), name in zip(face_locations, face_names):

		# 	top *= 5
		# 	right *= 5
		# 	bottom *= 5
		# 	left *= 5

		# 	# cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
		# 	# cv2.rectangle(img, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
		# 	font = cv2.FONT_HERSHEY_DUPLEX
		# 	cv2.putText(img, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


	#------------------------------------------------------------------------

		ret, img = cap.read()
		frame_number += 1
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	    # rgb_frame = img[:, :, ::-1]

	    # face_locations = face_recognition.face_locations(rgb_frame)
	    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
		cv2.circle(img, (960, 720), 10, (0,0,255), 2)

		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

			face_gray = gray[y:y+h, x:x+w]
			face_color = img[y:y+h, x:x+w]
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img,'Face',(x,y), font, 0.5, (11,255,255), 2, cv2.LINE_AA)


			eyes = eye_cascade.detectMultiScale(face_gray)
			for (ex,ey,ew,eh) in eyes:
				eyesC = cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(eyesC,'Eye',(ex,ey), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

			smiles = smileCascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=3, minSize=(15,15))
			for (ex,ey,ew,eh) in smiles:
				smileC = cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(smileC,'Mouth',(ex,ey), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
	          
			nose = noseCascade.detectMultiScale(face_gray)
			for (ex,ey,ew,eh) in nose:
				noseC = cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(noseC,'Nose',(ex,ey), font, 0.5, (11,255,255), 2, cv2.LINE_AA)


	#------------------------------------------------------------------------


		timestamp = datetime.datetime.now()
		cv2.putText(img, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, img.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		
		if total > frameCount:
			motion = md.detect(gray)
			if motion is not None:
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(img, (minX, minY), (maxX, maxY),
					(0, 0, 255), 2)
		
		md.update(gray)
		total += 1

		with lock:
			outputFrame = img.copy()

def generate():
	
	global outputFrame, lock

	
	while True:
		
		with lock:
			
			if outputFrame is None:
				continue

			
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			
			if not flag:
				continue

		
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	
	app.run(debug=True,
		threaded=True, use_reloader=False)

vs.stop()