"""Raspberry Pi Face Recognition Treasure Box
Positive Image Capture Script
Copyright 2013 Tony DiCola 

Run this script to capture positive images for training the face recognizer.
"""
import glob
import os
import sys
import select

import cv2

import hardware
import config
import face

person_name = sys.argv[1]

# Prefix for positive training image filenames.
POSITIVE_FILE_PREFIX = 'positive_'


def is_letter_input(letter):
	# Utility function to check if a specific character is available on stdin.
	# Comparison is case insensitive.
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		return input_char.lower() == letter.lower()
	return False


if __name__ == '__main__':
	camera = config.get_camera()
	#box = hardware.Box()
	
	folder_path = os.path.join('/home/pi/faces/', person_name) 
	# Create the directory for positive training images if it doesn't exist.
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	# Find the largest ID of existing positive images.
	# Start new images after this ID value.
	#files = sorted(glob.glob(os.path.join(folder_path, 
	#	'[0-9][0-9][0-9].pgm')))
	
	#eventually add some logic to only find .pgm files
	file_names = []
	files = os.listdir(folder_path)
	count = 0
	if len(files) > 0:
		for file in files:
			file_names.append(int(file.replace('.pgm', '')))
		count = max(file_names)

	print 'Capturing positive training images.'
	print 'Press button or type c (and press enter) to capture an image.'
	print 'Press Ctrl-C to quit.'
	while True:
		# Check if button was pressed or 'c' was received, then capture image.
		#if box.is_button_up() or is_letter_input('c'):
		print 'Capturing image...'
		image = camera.read()
		# Convert image to grayscale.
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				
		# Get coordinates of single face in captured image.
		result = face.detect_face(image, single = True)
		if result is not None:
                        x, y, w, h = result
                        crop = face.crop(image, x, y, w, h)
                	# Save image to file.
                	filename = os.path.join(folder_path, '%s.pgm' % (count))

                       	cv2.imwrite(filename, crop)
                	print 'Found face and wrote training image', filename
                	count += 1
			cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0))

                cv2.imshow('Frame', image)
                cv2.waitKey(1) & 0xFF

		
