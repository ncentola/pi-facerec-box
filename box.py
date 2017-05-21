#"""Raspberry Pi Face Recognition Treasure Box
#Treasure Box Script
#Copyright 2013 Tony DiCola 
#"""
import cv2
import config
import face
import hardware
from sqlalchemy import create_engine
import pandas as pd
import pigpio
import RPi.GPIO as GPIO
import time
import datetime

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#GPIO.setup(22, GPIO.OUT)
pi = pigpio.pi()
#p = GPIO.PWM(22, 50)
#p.start(7.5)

def log(engine, user_id, confidence):
	row = [(user_id, confidence, datetime.datetime.now())]
	df = pd.DataFrame.from_records(row, columns = ['user_id', 'confidence', 'created_at'])
	df.to_sql('activity', engine, if_exists = 'append')
	
def dispense(name):
        print 'Dispensing for ' + name
        #GPIO.output(22, 1)
        #time.sleep(0.0015)
        #GPIO.output(22, 0)
        #time.sleep(2)
        pi.set_servo_pulsewidth(22, 1000)
        time.sleep(0.25) # sleep 1 second
        pi.set_servo_pulsewidth(22, 2000)
        time.sleep(0.25) # sleep 1 second

def is_button_pressed():
	return GPIO.input(18)

if __name__ == '__main__':
	# Load training data into model
	print 'Loading training data...'
	model = cv2.createEigenFaceRecognizer()
	model.load(config.TRAINING_FILE)
	print 'Training data loaded!'
	
	# Initialize camer and box.
	camera = config.get_camera()
	
	# read in users lookup table
	engine = create_engine('postgresql://root@localhost:5432/pi')
	users = pd.read_sql('users', engine)

	while True:
		try:
			image = camera.read()
			# Convert image to grayscale.
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		
			# Get coordinates of single face in captured image.
			faces = face.detect_face(image, single = False)
			pi.write(25, 1)
			if faces is not None:
                        	pi.write(25, 0)
				for (x, y, w, h) in faces:
                                	cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0))


			if faces is not None:
			
				for facez in faces:
					x, y, w, h = facez
					
					## Crop and resize image to face.
					crop = face.resize(face.crop(image, x, y, w, h))
				
					## Test face against model.
					user_id, confidence = model.predict(crop)
				
					name = users['name'].loc[users['user_id'] == user_id]
					print name
					print type(name)
					name = str(name)
					#print name.at[0,1]
					cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0))
					cv2.putText(image, name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
				
					print(is_button_pressed())
					if not is_button_pressed():
						log(engine, user_id, confidence)
						dispense(name)
						

					print name
					print confidence
				
                	cv2.imshow('Frame', image)
                	cv2.waitKey(1) & 0xFF

		except KeyboardInterrupt:
			pi.stop()
			GPIO.cleanup()

	
		#print 'Predicted {0} face with confidence {1} (lower is more confident).'.format(
		#	'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', 
		#	confidence)
		#if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
		#	print 'Recognized face!'
		#	box.unlock()
		#else:
		#	print 'Did not recognize face!'
