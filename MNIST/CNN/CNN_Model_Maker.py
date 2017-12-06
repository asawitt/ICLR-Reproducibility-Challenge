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


#Load MNIST dataset
input_directory = "../datasets/ULN2500/"
training_filename_x = input_directory + "MNIST_uniform_label_noise_0_x"
training_filename_y = input_directory + "MNIST_uniform_label_noise_0_y"

test_filename_x = "../datasets/MNIST_raw/mnist_test_x"
test_filename_y = "../datasets/MNIST_raw/mnist_test_y"

labels = [[1 if i == j else 0 for i in range(10)] for j in range(10)]
batch_size = 200
num_epochs = 2

def str_to_img(s):
	img = list(map(lambda x: 1 if x=='1' else 0,s.split(",")))
	return img

def shape(line):
	return np.array(list(map(lambda x: int(x), line.split(",")))).reshape(-1,28)

def make_model():
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
	optimizer=keras.optimizers.Adadelta(lr=0.1),
	metrics=['accuracy'])
	
	return model

def get_file_data(filename_x,filename_y,one_hot_labels=True):
	with open(filename_x) as file_x,open(filename_y) as file_y:
		r_labels = []; r_data = [];
		for line in file_x:
			r_data.append(shape(line))
		if one_hot_labels:
			for line in file_y:
				r_labels.append(labels[int(line.strip())])
			r_labels=np.array(r_labels)
		else:
			for line in file_y:
				r_labels.append(int(line.strip()))
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
	train_data,train_labels = get_file_data(training_filename_x,training_filename_y)
	model = make_model()
	model.fit(train_data,train_labels,epochs=num_epochs,batch_size=batch_size)
	model.save('CNN_ULN_0.h5')
	# test_data,test_labels = get_file_data(test_filename_x,test_filename_y,False)
	# accuracy = get_test_accuracy(test_data,test_labels,model)
	# print(accuracy)



if __name__ == "__main__":
	main()

