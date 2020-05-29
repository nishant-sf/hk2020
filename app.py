# Importing required libraries
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import placeholder
from tensorflow.python.keras.models import Sequential, Model, model_from_json
import matplotlib.pyplot as plt
import keras
import pickle
import wave  # !pip install wave
import os
import pandas as pd
import numpy as np
import sys
import warnings
import librosa
import librosa.display

# from OpenSSL import SSL
# # context = SSL.Context(SSL.OP_NO_SSLv3)
# # context.use_privatekey_file('server.key')
# # context.use_certificate_file('server.crt')

from flask import Flask
app = Flask(__name__)


def speedNpitch(data):
	"""
	Speed and Pitch Tuning.
	"""
	# you can change low and high here
	length_change = np.random.uniform(low=0.8, high=1)
	speed_fac = 1.2 / length_change  # try changing 1.0 to 2.0 ... =D
	tmp = np.interp(np.arange(0, len(data), speed_fac),
					np.arange(0, len(data)), data)
	minlen = min(data.shape[0], tmp.shape[0])
	data *= 0
	data[0:minlen] = tmp[0:minlen]
	return data


def prepare_data(file_path, aug, mfcc, sampling_rate, audio_duration):
	n_mfcc = 30
	n_melspec = 60
	n = n_mfcc
	if mfcc == 0:
		n = n_melspec
	X = np.empty(shape=(1, n, 216, 1))
	input_length = sampling_rate * audio_duration

	data, _ = librosa.load(file_path, sr=sampling_rate,
						   res_type="kaiser_fast", duration=2.5, offset=0.5)

	# Random offset / Padding
	if len(data) > input_length:
		max_offset = len(data) - input_length
		offset = np.random.randint(max_offset)
		data = data[offset:(input_length+offset)]
	else:
		if input_length > len(data):
			max_offset = input_length - len(data)
			offset = np.random.randint(max_offset)
		else:
			offset = 0
		data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

	# Augmentation? 
	if aug == 1:
		data = speedNpitch(data)

	# which feature?
	if mfcc == 1:
		# MFCC extraction 
#         MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
		MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
		MFCC = np.expand_dims(MFCC, axis=-1)
		X[0,] = MFCC

	else:
		# Log-melspectogram
		melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)   
		logspec = librosa.amplitude_to_db(melspec)
		logspec = np.expand_dims(logspec, axis=-1)
		X[0,] = logspec
	
	return X

def score(predvec):

	# ['female_angry' 'female_disgust' 'female_fear' 'female_happy' 
	# 'female_neutral' 'female_sad' 'female_surprise' 

	# 'male_angry' 'male_disgust' 'male_fear' 'male_happy' 
	# 'male_neutral' 'male_sad' 'male_surprise']

	# 0/1/2/5, 7/8/9/12 -> 8/14 -> 4/7
	# 4, 11             -> 2/14 -> 1/7
	# 3/6, 10/13        -> 4/14 -> 2/7

	# print(predvec)
	# print(type(predvec))
	# print(predvec.shape)
	predvec = predvec[0]
	hi_idx = [3,6,10,13]
	med_idx = [4,11]
	lo_idx = [0,1,2,5,7,8,9,12]
	
	max_idx = np.argmax(predvec)
	if max_idx in hi_idx:
		# high
		sum_hi = predvec[hi_idx].sum()
		return 60 + sum_hi * 40
	elif max_idx in med_idx:
		# medium
		sum_med = predvec[med_idx].sum()
		return 40 + sum_med * 20
	else:
		# low
		sum_lo = predvec[lo_idx].sum()
		return sum_lo * 40

@app.route('/scorer/<filename>')
def scorer(filename=None):
	json_file = open('input/model2_json.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	tf.disable_v2_behavior()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights("input/VoiceModel2.h5")
	print("Loaded model from disk")

	# the optimiser
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	audio_file_name = 'input/' + filename # + '3.wav'

	X = prepare_data(audio_file_name, 
					 aug=1, mfcc=1, sampling_rate=44100, audio_duration=2.5)

	newpred = loaded_model.predict(X, 
							 batch_size=16, 
							 verbose=1)

	return str(score(newpred))

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    context = ('local.crt', 'local.key')#certificate and key files
    app.run(threaded=True, port = int(os.environ.get('PORT', 5000)), ssl_context='adhoc') #context)
