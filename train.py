"""Raspberry Pi Face Recognition Treasure Box
Face Recognition Training Script
Copyright 2013 Tony DiCola 

Run this script to train the face recognition system with positive and negative
training images.  The face recognition model is based on the eigen faces
algorithm implemented in OpenCV.  You can find more details on the algorithm
and face recognition here:
  http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
"""
import fnmatch
import os

import cv2
import numpy as np
import datetime
import config
import face

import pandas as pd
from sqlalchemy import create_engine


MEAN_FILE = 'mean.png'
POSITIVE_EIGENFACE_FILE = 'positive_eigenface.png'
NEGATIVE_EIGENFACE_FILE = 'negative_eigenface.png'


def walk_files(directory, match='*'):
	"""Generator function to iterate through all files in a directory recursively
	which match the given filename match parameter.
	"""
	for root, dirs, files in os.walk(directory):
		for filename in fnmatch.filter(files, match):
			yield os.path.join(root, filename)

def prepare_image(filename):
	"""Read an image as grayscale and resize it to the appropriate size for
	training the face recognition model.
	"""
	return face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

def normalize(X, low, high, dtype=None):
	"""Normalizes a given array in X to a value between low and high.
	Adapted from python OpenCV face recognition example at:
	  https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
	"""
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	# normalize to [0...1].
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

if __name__ == '__main__':
	print "Reading training images..."
	faces = []
	labels = []
	faces_count = 0

	dir_id = 0
	id_name_lookup = []	
	
	engine = create_engine('postgresql://root@localhost:5432/pi')	
	#query to find all people not already in users
	if engine.dialect.has_table(engine, 'users'):
		users = pd.read_sql('users', engine)	
		dir_id = users['user_id'].max() + 1
		user_names = list(pd.Series(users['name']))
	else:
		dir_id = 0
		user_names = []
		dummy_data = [(9999, 'blah', 0)]
		users = pd.DataFrame.from_records(dummy_data, columns = ['user_id', 'name', 'created_at'])

	# Read all faces
	for root, dirs, files in os.walk(config.FACES_DIR):
		for dir in dirs:
			if dir in user_names:
				print 'test'
				existing_id = int(users['user_id'].loc[users['name'] == dir])
				id_name_lookup.append((existing_id, dir, datetime.datetime.now()))
				for file in os.listdir(os.path.join(root, dir)):
                                	filename = os.path.join(root, dir, file)
                                	faces.append(prepare_image(filename))
                                	labels.append(existing_id)
                                	faces_count = faces_count + 1

			else:
				id_name_lookup.append((dir_id, dir, datetime.datetime.now())) 
				for file in os.listdir(os.path.join(root, dir)):
					filename = os.path.join(root, dir, file)
					faces.append(prepare_image(filename))
					labels.append(dir_id)
					faces_count = faces_count + 1
				dir_id = dir_id + 1
	

	df = pd.DataFrame.from_records(id_name_lookup, columns = ['user_id', 'name', 'created_at'])
	df_new = df.loc[~df['name'].isin(users['name'])]
	print 'Writing ' + str(len(df_new.index)) + ' rows.'
	df_new.to_sql('users', engine, if_exists = 'append')
	
	print 'Read', faces_count, 'images.'

	# Train model
	print 'Training model...'
	model = cv2.createEigenFaceRecognizer()
	model.train(np.asarray(faces), np.asarray(labels))

	# Save model results
	model.save(config.TRAINING_FILE)
	print 'Training data saved to', config.TRAINING_FILE

	# Save mean and eignface images which summarize the face recognition model.
	mean = model.getMat("mean").reshape(faces[0].shape)
	cv2.imwrite(MEAN_FILE, normalize(mean, 0, 255, dtype=np.uint8))
	eigenvectors = model.getMat("eigenvectors")
	pos_eigenvector = eigenvectors[:,0].reshape(faces[0].shape)
	cv2.imwrite(POSITIVE_EIGENFACE_FILE, normalize(pos_eigenvector, 0, 255, dtype=np.uint8))
	#neg_eigenvector = eigenvectors[:,1].reshape(faces[0].shape)
	#cv2.imwrite(NEGATIVE_EIGENFACE_FILE, normalize(neg_eigenvector, 0, 255, dtype=np.uint8))
