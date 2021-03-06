from mnist import MNIST
import numpy as np
import scipy.misc 
import random
import os

NUM_IMAGES = 60000
alphas = [0,10,20,30,40,50,60,70,80,90,100]
output_dir = "datasets/ULN" + str(NUM_IMAGES) + "/"
os.makedirs(output_dir)
output_filename = output_dir + "MNIST_uniform_label_noise_"


label_space = ['0','1','2','3','4','5','6','7','8','9']
#Load MNIST dataset
mnist_directory = "datasets/MNIST_raw"
mndata = MNIST(mnist_directory)
images, labels = mndata.load_training()

output_x_files = {}
output_y_files = {}

BINARIZATION_THRESHOLD = 150

def img_to_str(img):
	img_str = "".join(list(map(lambda x: '1,' if x > BINARIZATION_THRESHOLD else '0,',img)))
	return img_str[:-1]	

def str_to_img(s):
	img = list(map(lambda x: 1 if x=='1' else 0,s.split(",")))
	return img

def main():
	#Open all the files
	for alpha in alphas:
		output_x_files[alpha] = open(output_filename + str(alpha) + "_x", 'w')
		output_y_files[alpha] = open(output_filename + str(alpha) + "_y", 'w')

	global NUM_IMAGES
	NUM_IMAGES = min(len(images),NUM_IMAGES) #in case NUM_IMAGES is too high
	#For each image we rewrite it to a second file, while throwing in variable number of incorrect labels
	index = 0
	for image,label in zip(images[0:NUM_IMAGES],labels[0:NUM_IMAGES]):
		if not index%250:
			print(str(index) + "/" + str(NUM_IMAGES))
		index += 1
		img_str = img_to_str(image)
		label = str(label)
		for alpha in alphas:
			#Write correct label
			output_x_files[alpha].write(img_str + "\n")
			output_y_files[alpha].write(label + "\n")
			# Write incorrect label(s)
			for i in range(alpha):
				f_label = random.choice(label_space)
				output_x_files[alpha].write(img_str + "\n")
				output_y_files[alpha].write(f_label + "\n")


	#Close all the files
	for file in list(output_x_files.values()) + list(output_y_files.values()):
		file.close()

if __name__ == "__main__":
	main()
