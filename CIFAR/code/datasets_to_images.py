# -*- coding: utf-8 -*-
from numpy.random import seed as np_seed
np_seed(9)
import random
random.seed(9)

import numpy
import pickle

import scipy
import os

def channel_last_reshape(im):
    by_column = im.reshape(3, 1024).T
    result = numpy.zeros((32,32,3))
    count = 0
    for i in range(32):
        for j in range(32):
            result[i,j] = by_column[count]
            count+=1
    return result

def write_polluted_with_cifar_100(train10_folder, base_name10, cifar100, max_size, alpha_list, base_alpha_folder):
    
# Code to generate cifar-10 polluted with cifar-100    
    '''
    1. Load 2 cifar 10 batches with labels and merge them.
    2. Load cifar 100 training.
    3. Select max_size from cifar 10 (12k)
    4. Build alpha_list. alpha = 0 IS ALREADY DONE!!!!!
    5. Reshape max_size first images from cifar 10 and cifar 100.
    6. For alpha in alpha_list:
           a. Accumulate max_size images from cifar 10.
           b. For each max_size images in cifar 100:
                  Assign image to alpha diff categories at random.
           c. Write images as they are labeled into the folder for alpha.
    '''
    
    batch1 = pickle.load(open(train10_folder+base_name10+'1', 'rb'), encoding='bytes')
    batch2 = pickle.load(open(train10_folder+base_name10+'2', 'rb'), encoding='bytes')
    batch_c100 = pickle.load(open(cifar100, 'rb'), encoding='bytes')
    
#    max_size = 12000
    
    c10_labels = numpy.concatenate((batch1[b'labels'], batch2[b'labels']), axis = 0)[0:max_size]
    c10_data = numpy.concatenate((batch1[b'data'], batch2[b'data']), axis = 0)[0:max_size]
    data_c100 = batch_c100[b'data']
    c10_data = numpy.array(list(map(channel_last_reshape, c10_data)))
    data_c100 = numpy.array(list(map(channel_last_reshape, data_c100)))
    
    os.mkdir(base_alpha_folder)
    
    for alpha in alpha_list:
        count = 0
        current_alpha_folder = base_alpha_folder+str(alpha)+'/'
        os.mkdir(current_alpha_folder)
        for i in range(0,10):
            os.mkdir(current_alpha_folder+str(i))
        for i in range(len(c10_data)):
            scipy.misc.imsave(current_alpha_folder+str(c10_labels[i])+'/image_'+str(count)+'.png', c10_data[i] )
            count+=1
        max_from_c100 = alpha*max_size
        for i in range(max_from_c100):
            index = i%len(data_c100)
            rand_class = random.choice(list(range(0,10)))
            scipy.misc.imsave(current_alpha_folder+str(rand_class)+'/image_'+str(count)+'.png', data_c100[index] )
            count+=1    

def write_cifar_10_polluted_with_itself(train10_folder, base_name10, test_file, max_size, alpha_list, base_alpha_folder):
   
    batch1 = pickle.load(open(train10_folder+base_name10+'1', 'rb'), encoding='bytes')
    batch2 = pickle.load(open(train10_folder+base_name10+'2', 'rb'), encoding='bytes')
    test_size = int(max_size/6)
    test_batch = pickle.load(open(train10_folder+test_file, 'rb'), encoding='bytes')
     
    c10_labels = numpy.concatenate((batch1[b'labels'], batch2[b'labels']), axis = 0)[0:max_size]
    c10_data = numpy.concatenate((batch1[b'data'], batch2[b'data']), axis = 0)[0:max_size] 
    c10_data = numpy.array(list(map(channel_last_reshape, c10_data)))
    
    test_labels = test_batch[b'labels'][0:test_size]
    test_data = test_batch[b'data'][0:test_size]
    test_data = numpy.array(list(map(channel_last_reshape, test_data)))
    
    base_train = base_alpha_folder+'train/'
    base_test = base_alpha_folder+'test/'
    
    os.mkdir(base_alpha_folder)
    os.mkdir(base_train)
    os.mkdir(base_test)
    
    classes = list(range(0,10))
    
    for alpha in alpha_list:
        count = 0
        os.mkdir(base_train+str(alpha))
        for i in range(0,10):
            os.mkdir(base_train+str(alpha)+'/'+str(i))
        for index in range(0, max_size):
            true_class = c10_labels[index]
            image = c10_data[index]
            scipy.misc.imsave(base_train+str(alpha)+'/'+str(true_class)+'/image_'+str(count)+'.png', image)
            count+=1
            classes.remove(true_class)
            bad_labels = random.choices(classes, k = alpha)
            for label in bad_labels:
                scipy.misc.imsave(base_train+str(alpha)+'/'+str(label)+'/image_'+str(count)+'.png', image)
                count+=1
            classes.append(true_class)
    
    for i in range(0,10):
        os.mkdir(base_test+str(i))
    for index in range(0, test_size):
        count = 0
        label = test_labels[index]
        data = test_data[index]
        scipy.misc.imsave(base_test+str(label)+'/image_'+str(count)+'.png', data)
        count+=1
if __name__ == '__main__':
    
    WRITE_POLLUTED_BY_ITSELF = True
    WRITE_POLLUTED_BY_CIFAR_100 = True
    
    if WRITE_POLLUTED_BY_CIFAR_100:
        train10_folder = '../data/cifar-10-batches-py/'
        base_name10 = 'data_batch_'
        cifar100 = '../data/cifar-100-python/train'
        max_size = 12000
        alpha_list = [0,2,4,6,8]
        base_alpha_folder = '../data/cifar-100_pollution/'    
        write_polluted_with_cifar_100(train10_folder, base_name10, cifar100, max_size, alpha_list, base_alpha_folder)
    
    if WRITE_POLLUTED_BY_ITSELF:
        train10_folder = '../data/cifar-10-batches-py/'
        base_name10 = 'data_batch_'
        test_file = 'test_batch'
        max_size = 12000
        alpha_list = [0,2,4,6,8]
        base_alpha_folder = '../data/alpha/'
        test_batch = write_cifar_10_polluted_with_itself(train10_folder, base_name10, test_file, max_size, alpha_list, base_alpha_folder)
   