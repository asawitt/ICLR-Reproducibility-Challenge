# -*- coding: utf-8 -*-
from numpy.random import seed
seed(9)
from tensorflow import set_random_seed
set_random_seed(9)

import numpy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.optimizers import Adadelta

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

import pickle
import time
import os

from nets.utilities import train_results, image_train_results

def test_net(input_shape, output_size, lr):
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape=input_shape, activation='relu', kernel_regularizer=l2()) )    
    model.add(Dropout(0.5))
    model.add(Conv2D(20, (5,5), activation='relu', kernel_regularizer=l2()) ) 
    model.add(Dropout(0.5))
    model.add(Conv2D(15, (5,5), activation='relu', kernel_regularizer=l2()) ) 
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units = output_size, activation='softmax'))
    model.compile(Adadelta(lr=lr), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def try_alphas(train_top_folder, test_folder, alpha_list, lr_map, base_save_folder, net_name):
    all_accs = {}
    for alpha in alpha_list:
        print('alpha = '+str(alpha)+' with lr = '+str(lr_map[alpha]))
        save_dir = base_save_folder+str(alpha)+'/'
        save_array = [10, save_dir, net_name+'-alpha_'+str(alpha)]
        data_folders = [train_top_folder+str(alpha)+'/', test_folder]
        current_net = test_net((32,32,3), 10, lr_map[alpha])
        results = image_train_results(current_net, 
                                     data_folders, 
                                     save_array, 
                                     num_epochs = 80, 
                                     test_freq = 2, 
                                     sleep_time = 5, 
                                     batch_size = 128, 
                                     class_weight = None, 
                                     target_size = (32,32), 
                                     augment = True,
                                     verbose = 0,
                                     get_train_acc = False)
        all_accs[alpha] = {'lr':lr_map[alpha], 'results':results}
        print('\n')
    return all_accs

def try_alphas_fix_lr(train_top_folder, test_folder, alpha_list, lr, base_save_folder, net_name):
    all_accs = {}
    for alpha in alpha_list:
        print('alpha = '+str(alpha)+' with lr = '+str(lr))
        save_dir = base_save_folder+str(alpha)+'/'
        save_array = [10, save_dir, net_name+'-alpha_'+str(alpha)]
        data_folders = [train_top_folder+str(alpha)+'/', test_folder]
        current_net = test_net((32,32,3), 10, lr)
        results = image_train_results(current_net, 
                                     data_folders, 
                                     save_array, 
                                     num_epochs = 80, 
                                     test_freq = 2, 
                                     sleep_time = 5, 
                                     batch_size = 128, 
                                     class_weight = None, 
                                     target_size = (32,32), 
                                     augment = True,
                                     verbose = 0,
                                     get_train_acc = False)
        all_accs[alpha] = {'lr':lr_map[alpha], 'results':results}
        print('\n')
    return all_accs
if __name__ == '__main__':
    CIFAR_10_BEST_LR = True
    CIFAR_100_POLLUTION = True
    CIFAR_10_VARY_LR = True
    
    test_folder = '../data/alpha/test/'
    alpha_list = [0,2,4,6,8]
    
    #Create folders to save nets
    #Create folder to save results
    '''
    .
        code
        data
        results
            cifar-10_self_pollute
                alpha1
                alpha2
                ...
            cifar-100_pollution
                alpha1
                alpha2
                ...
            cifar-10_vary_lr
                alpha1
                alpha2
                ...
    '''
    os.mkdir('../results')
    os.mkdir('../nets')
    
    os.mkdir('../results/cifar-10_self_pollute')
    os.mkdir('../results/cifar-100_pollution')
    os.mkdir('../results/cifar-10_vary_lr')
    
    os.mkdir('../nets/cifar-10_self_pollute')
    os.mkdir('../nets/cifar-100_pollution')
    os.mkdir('../nets/cifar-10_vary_lr')
    
    for alpha in alpha_list:
        os.mkdir('../nets/cifar-10_self_pollute/'+str(alpha))
        os.mkdir('../nets/cifar-100_pollution/'+str(alpha))
        os.mkdir('../nets/cifar-10_vary_lr/'+str(alpha))
    
    if CIFAR_10_BEST_LR:
        lr_map = {0:0.5, 2:0.5, 4:0.1, 6:0.1, 8:0.05, 10:0.1}
        train_top_folder = '../data/alpha/train/'
        base_save_folder = '../nets/cifar-10_self_pollute/'
        results_folder = '../results/cifar-10_self_pollute/'
        net_name = 'cifar-10_self_polluted'
        results = try_alphas(train_top_folder, 
                             test_folder, 
                             alpha_list, 
                             lr_map, 
                             base_save_folder, 
                             net_name)
        pickle.dump(results, open(results_folder+'accuracies.pickle', 'wb'))
    if CIFAR_100_POLLUTION:
        lr_map = {0:0.5, 2:0.5, 4:0.1, 6:0.1, 8:0.05, 10:0.1}
        train_top_folder = '../data/cifar-100_pollution/'
        base_save_folder = '../nets/cifar-100_pollution/'
        results_folder = '../results/cifar-100_pollution/'
        net_name = 'cifar-10_polluted_with_cifar-100'
        results = try_alphas(train_top_folder, 
                             test_folder, 
                             alpha_list, 
                             lr_map, 
                             base_save_folder, 
                             net_name)
        pickle.dump(results, open(results_folder+'accuracies.pickle', 'wb'))
    if CIFAR_10_VARY_LR:
        lrs = [0.1, 0.5]
        train_top_folder = '../data/alpha/train/'
        base_save_folder = '../nets/cifar-10_vary_lr/'
        results_folder = '../results/cifar-10_vary_lr/'
        results = {}
        for lr in lrs:
            net_name = 'cifar-10_vary_lr_lr:'+str(lr)
            res = try_alphas_fix_lr(train_top_folder, 
                                    test_folder, 
                                    alpha_list, 
                                    lr, 
                                    base_save_folder, 
                                    net_name)
            results[lr] = res
        pickle.dump(results, open(results_folder+'accuracies.pickle', 'wb'))
