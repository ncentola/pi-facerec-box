#"""Raspberry Pi Face Recognition Treasure Box
#Treasure Box Script
#Copyright 2013 Tony DiCola 
#"""
import cv2
import csv
import config
import face
import hardware


if __name__ == '__main__':
	# Load training data into model
	print 'Loading training data...'
	model = cv2.createEigenFaceRecognizer()
	model.load(config.TRAINING_FILE)
	print 'Training data loaded!'
	# Initialize camer and box.
	camera = config.get_camera()
	# read in id/label lookup table
	x = open('id_name_lookup.csv')
	id_name_lookup = csv.reader(x)
	print id_name_lookup
	
	while True:
		image = camera.read()
		# Convert image to grayscale.
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		# Get coordinates of single face in captured image.
		faces = face.detect_face(image, single = False)
		#if faces is not None:
        	#        for (x, y, w, h) in faces:
		#		cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0))
		#if faces is not None:
		#	print len(faces)
		#else:
		#	print 'no faces'
		cv2.imshow('Frame', image)
                cv2.waitKey(1) & 0xFF

		if faces is not None:
			for (x, y, w, h) in faces:
        	               	cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0))

		if faces is None:
			print 'Could not detect single face!  Check the image in capture.pgm' \
				  ' to see what was captured and try again with only one face visible.'
			continue
		else:
			for facez in faces:
				x, y, w, h = facez
				## Crop and resize image to face.
				crop = face.resize(face.crop(image, x, y, w, h))
				## Test face against model.
		
				label, confidence = model.predict(crop)
				print label
				print confidence


	
		#print 'Predicted {0} face with confidence {1} (lower is more confident).'.format(
		#	'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', 
		#	confidence)
		#if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
		#	print 'Recognized face!'
		#	box.unlock()
		#else:
		#	print 'Did not recognize face!'
