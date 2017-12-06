from mnist import MNIST
import numpy as np
import scipy.misc 
import random

#Load MNIST dataset
mnist_directory = "datasets/MNIST_raw"
mndata = MNIST(mnist_directory)
images, labels = mndata.load_testing()

output_filename_x = mnist_directory + "/mnist_test_x"
output_filename_y = mnist_directory + "/mnist_test_y"

BINARIZATION_THRESHOLD = 150 

def img_to_str(img):
	img_str = "".join(list(map(lambda x: '1,' if x > BINARIZATION_THRESHOLD else '0,',img)))
	return img_str[:-1]	\

def main():
	with open(output_filename_x,'w') as out_x, open(output_filename_y,'w') as out_y:
		for image,label in zip(images,labels):
			img_str = img_to_str(image)
			label = str(label)
			out_x.write(img_str + "\n")
			out_y.write(label + "\n")


if __name__ == "__main__":
	main()
