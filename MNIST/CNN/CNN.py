from mnist import MNIST
import numpy as np
import scipy.misc 

#Load MNIST dataset
mnist_directory = "../datasets"
mndata = MNIST(mnist_directory)
images, labels = mndata.load_training()

scipy.misc.imshow(np.array(images[0]).reshape(-1,28))
