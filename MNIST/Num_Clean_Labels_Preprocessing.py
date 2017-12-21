from mnist import MNIST
import numpy as np
import scipy.misc 
import random
import os

num_images = [100,500,1000,2000,3000,4000,5000,10000,20000,30000,40000,50000,60000]
alphas = [0,10,20,50]
output_dir = "datasets/NUM_CLEAN_LABELS/"
os.makedirs(output_dir)
output_filename = output_dir + "MNIST_num_clean_labels_"


label_space = ['0','1','2','3','4','5','6','7','8','9']
#Load MNIST dataset
mnist_directory = "datasets/MNIST_raw"
mndata = MNIST(mnist_directory)
images, labels = mndata.load_training()

output_x_files = [{} for i in range(len(num_images))]
output_y_files = [{} for i in range(len(num_images))]

BINARIZATION_THRESHOLD = 150

def img_to_str(img):
	img_str = "".join(list(map(lambda x: '1,' if x > BINARIZATION_THRESHOLD else '0,',img)))
	return img_str[:-1]	

def str_to_img(s):
	img = list(map(lambda x: 1 if x=='1' else 0,s.split(",")))
	return img

def main():
	global num_images

	#Open all the files
	for n in range(len(num_images)):
		for alpha in alphas:
			output_x_files[n][alpha] = open(output_filename + str(num_images[n]) + "_" + str(alpha) + "_x", 'w')
			output_y_files[n][alpha] = open(output_filename + str(num_images[n]) + "_" + str(alpha) + "_y", 'w')

	for n in range(len(num_images)):
		num_imgs = min(len(images),num_images[n]) #in case n is too high
		#For each image we rewrite it to a second file, while throwing in variable number of incorrect labels
		index = 0
		for image,label in zip(images[0:num_imgs],labels[0:num_imgs]):
			if not index%250:
				print(str(index) + "/" + str(num_imgs))
			index += 1
			img_str = img_to_str(image)
			label = str(label)
			for alpha in alphas:
				#Write correct label
				output_x_files[n][alpha].write(img_str + "\n")
				output_y_files[n][alpha].write(label + "\n")
				# Write incorrect label(s)
				for i in range(alpha):
					f_label = random.choice(label_space)
					output_x_files[n][alpha].write(img_str + "\n")
					output_y_files[n][alpha].write(f_label + "\n")


	#Close all the files
	for f in output_x_files + output_y_files:
		for file in list(f.values()):
			file.close()

if __name__ == "__main__":
	main()
