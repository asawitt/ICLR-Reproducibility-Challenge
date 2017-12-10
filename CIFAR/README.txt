_____Steps to run the CIFAR-10 experiments_____

1. Load the CIFAR-10 and CIFAR-100 datasets using these links:

CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 
CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

and decompress them inside the folder 'data'. The folder containing
CIFAR-10 data should be named 'cifar-10-batches-py' and the folder 
for CIFAR-100 should be named 'cifar-100-python'.

2. From the folder 'code', run the file 'datasets_to_images.py'.
This will create the datasets to train the nets on.

3. From the folder 'code', run the file 'cifar_nets.py'. This will
train the nets with varying alpha and learning rates. The nets will
be saved in the new folder 'nets' and the experiments accuracies
on the test set will be saved as pickled objects in the folder
new 'results'.
