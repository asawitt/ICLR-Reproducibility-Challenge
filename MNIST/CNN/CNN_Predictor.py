from keras.models import load_model
import numpy as np



test_filename_x = "../datasets/MNIST_raw/mnist_test_x"
test_filename_y = "../datasets/MNIST_raw/mnist_test_y"

def shape(line):
	return np.array(list(map(lambda x: int(x), line.split(",")))).reshape(-1,28)

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
	test_data,test_labels = get_file_data(test_filename_x,test_filename_y,False)
	model_filepath = "Models/CNN_ULN2000_50.h5"
	model = load_model(model_filepath)
	accuracy = get_test_accuracy(test_data,test_labels,model)
	print(accuracy)


if __name__=="__main__":
	main()