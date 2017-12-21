from mnist import MNIST
import numpy as np
import scipy.misc 

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import optimizers,losses
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
import keras
import math

import gc


alphas = [0,10,20,30,40,50,60,70,80,90,100]
#Load MNIST dataset
NUM_DATAPOINTS = 60000

input_directory = "../datasets/ULN" + str(NUM_DATAPOINTS) + "/"
training_filename_x = input_directory + "MNIST_uniform_label_noise_"
training_filename_y = input_directory + "MNIST_uniform_label_noise_"

test_filename_x = "../datasets/MNIST_raw/mnist_test_x"
test_filename_y = "../datasets/MNIST_raw/mnist_test_y"

labels = [[1 if i == j else 0 for i in range(10)] for j in range(10)]
batch_size = 500
num_epochs = 5


def str_to_img(s):
	img = list(map(lambda x: 1 if x=='1' else 0,s.split(",")))
	return img

def shape(line):
	return np.array(list(map(lambda x: int(x), line.split(",")))).reshape(-1,28)

def make_model(alpha):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=(28,28,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=keras.optimizers.Adadelta(lr=.1, rho=0.95, epsilon=1e-6),
	metrics=['accuracy'])
	
	return model

def get_file_data(filename_x,filename_y,one_hot_labels=True,start_line=0, num_lines=0):

	with open(filename_x) as file_x,open(filename_y) as file_y:
		for i in range(start_line):
			next(file_x)
			next(file_y)
		r_labels = []; r_data = [];

		#DATA
		num_left = num_lines
		for line in file_x:
			r_data.append(shape(line))
			num_left -= 1
			if not num_left:
				break

		#ONE-HOT LABELS
		num_left = num_lines
		if one_hot_labels:
			for line in file_y:
				r_labels.append(labels[int(line.strip())])
				num_left -= 1
				if not num_left:
					break
			r_labels=np.array(r_labels)
		
		else:
			for line in file_y:
				r_labels.append(int(line.strip()))
		
	gc.collect()
	if num_lines:
		r_data = np.array(r_data).reshape(num_lines,28,28,1)
	else:
		r_data = np.array(r_data).reshape(len(r_data),28,28,1)

	return r_data,r_labels

def get_test_accuracy(test_data,test_labels,model):
	predicted_labels = model.predict(test_data)
	predicted_labels= list(map(lambda x: np.argmax(x),predicted_labels))
	num_right = 0
	for predicted,actual in zip(predicted_labels,test_labels):
		num_right += 1 if predicted == actual else 0
	return num_right/len(test_labels)


def main():
	test_data,test_labels = get_file_data(test_filename_x,test_filename_y,False)
	highest_accuracy = {}
	for alpha in alphas:
		model = make_model(alpha)
		highest_accuracy[alpha] = 0
		start_line = 0
		#Train in batches of 330000 images, since we don't have enough memory for larger sets
		for i in range(num_epochs):
			n_lines = alpha*NUM_DATAPOINTS + NUM_DATAPOINTS

			while (n_lines != 0):
				num_lines_in_train = min(n_lines,330000)
				train_data,train_labels = get_file_data(
					training_filename_x + str(alpha) + '_x',
					training_filename_y + str(alpha) + '_y',
					True,
					start_line,
					num_lines_in_train
				)
				n_lines -= num_lines_in_train
				model.fit(train_data,train_labels,epochs=1,batch_size=batch_size)
				#Display/Save test-set accuracy every epoch
			accuracy = get_test_accuracy(test_data,test_labels,model)
			if accuracy > highest_accuracy[alpha]:
				highest_accuracy[alpha] = accuracy		
			print(str(i) + ") " +  str(accuracy))
		
		model.save('Models/CNN_ULN' + str(NUM_DATAPOINTS) + "_" + str(alpha) + '.h5')
		#Test-set Accuracy
		with open("test.txt",'w') as results_file:
			for key in highest_accuracy.keys():
				results_file.write("highest_accuracy[" + str(key) + "]=" + str(highest_accuracy[key]))
		
if __name__ == "__main__":
	main()

