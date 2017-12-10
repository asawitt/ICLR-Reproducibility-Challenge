# -*- coding: utf-8 -*-

import time
import numpy
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

import sys

def save_classifier(classifier, model_file, model_weights):
    to_json = classifier.to_json()
    
    with open(model_file, 'w') as file:
        file.write(to_json)
        classifier.save_weights(model_weights)
        
def load_classifier(filename, weights_file):
    loaded_from_json = open(filename, 'r').read()
#    
    loaded_classifier = model_from_json(loaded_from_json)
    loaded_classifier.load_weights(weights_file)
    return loaded_classifier

def image_epoch_accs(classifier, max_epoch, train, test, train_steps, test_steps, test_freq, save_arr, sleep_time = 15, class_weight = None, verbose = 1, get_train_acc = True):    
    accuracies = []
    save_freq = save_arr[0]
    save_folder = save_arr[1]
    name = save_arr[2]
    assert max_epoch % save_freq == 0, 'Should have max_epoch divisible by save_freq: '+str(max_epoch)+', '+str(save_freq)
    model_file = save_folder+name+'.nn'
    to_json = classifier.to_json()
    max_acc = 0
    best_epoch = -1
    with open(model_file, 'w') as file:
        file.write(to_json)
    for index in range(max_epoch):
        sys.stdout.write('\rEpoch '+str(index+1)+'. Best accuracy: '+str(max_acc)+' at epoch '+str(best_epoch)+'.')
        classifier.fit_generator(train, epochs = 1, steps_per_epoch = train_steps, class_weight = class_weight, verbose=verbose)
        if (index+1) % test_freq == 0:    
            pair = [0,0,0]
            pair[0] = index+1
            if get_train_acc:
                pair[1] = classifier.evaluate_generator(train, steps = train_steps)[1]
            else:
                pair[1] = -1
            pair[2] = classifier.evaluate_generator(test, steps = test_steps)[1]
            if pair[2] > max_acc:
                max_acc = pair[2]
                best_epoch = index+1
            accuracies.append(pair)
        if (index+1) % save_freq == 0:
            weights_file =  save_folder+name+'-'+str(index+1)+'_ep.weights' 
            classifier.save_weights(weights_file)
        time.sleep(sleep_time)
    return accuracies
#Assumes classifier is compiled.
def image_train_results(classifier, data_folders, save_arr, num_epochs, test_freq, sleep_time = 15, batch_size = 128, class_weight = None, target_size = (64,64), augment = False, verbose = 1, get_train_acc = True):
    train_folder = data_folders[0]
    test_folder = data_folders[1]
    if augment:
        train_generator = ImageDataGenerator ( rescale=1./255, shear_range=0.2, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, rotation_range=20)#, samplewise_center=True, samplewise_std_normalization=True)
        test_generator = ImageDataGenerator(rescale=1./255)#, samplewise_center=True, samplewise_std_normalization=True)    
    else:
        train_generator = ImageDataGenerator( rescale=1./255) #, shear_range=0.2, zoom_range=0.2, samplewise_center=True, samplewise_std_normalization=True)
        test_generator = ImageDataGenerator(rescale=1./255)#, samplewise_center=True, samplewise_std_normalization=True)
    train = train_generator.flow_from_directory(directory=train_folder, 
                                                target_size=target_size, 
                                                class_mode='categorical', 
                                                color_mode='rgb', 
                                                batch_size=batch_size)
    test = test_generator.flow_from_directory(directory=test_folder, 
                                              target_size=target_size, 
                                              class_mode='categorical', 
                                              color_mode='rgb', 
                                              batch_size=batch_size)
    train_steps = len(train.filenames)
    test_steps = len(test.filenames)
    
    train_steps = int((train_steps+batch_size)/batch_size)
    test_steps = int((test_steps+batch_size)/batch_size)
    return image_epoch_accs(classifier, num_epochs, train, test, train_steps, test_steps, test_freq, save_arr, sleep_time, class_weight, verbose, get_train_acc)
def epoch_accs(classifier, max_epoch, train, test, batch_size, test_freq, save_arr, sleep_time = 15, class_weight = None):    
    accuracies = []
    save_freq = save_arr[0]
    save_folder = save_arr[1]
    name = save_arr[2]
    assert max_epoch % save_freq == 0, 'Should have max_epoch divisible by save_freq: '+str(max_epoch)+', '+str(save_freq)
    model_file = save_folder+name+'.nn'
    to_json = classifier.to_json()
    with open(model_file, 'w') as file:
        file.write(to_json)
    for index in range(max_epoch):
        print('Epoch '+str(index+1))
        classifier.fit(train[0] ,train[1], epochs = 1, batch_size=batch_size, class_weight = class_weight)
        if (index+1) % test_freq == 0:    
            pair = [0,0,0]
            pair[0] = index+1
            pair[1] = classifier.evaluate(train[0], train[1], batch_size = batch_size, verbose = 0)[1]
            pair[2] = classifier.evaluate(test[0], test[1], batch_size=batch_size, verbose = 0)[1]
            accuracies.append(pair)
        if (index+1) % save_freq == 0:
            weights_file =  save_folder+name+'-'+str(index+1)+'_ep.weights' 
            classifier.save_weights(weights_file)
        time.sleep(sleep_time)
    return accuracies

def train_results(classifier, data, save_arr, num_epochs, test_freq, sleep_time = 15, batch_size = 128, class_weight = None):
    train = data[0]
    test = data[1]
    train_steps = len(train[0])
    test_steps = len(test[0])
    train_steps = int((train_steps+batch_size)/batch_size)
    test_steps = int((test_steps+batch_size)/batch_size)
    return epoch_accs(classifier, num_epochs, train, test, batch_size, test_freq, save_arr, sleep_time, class_weight)

def cross_validated_accuracies(net_generator, inputs, outputs, parameters, cv = 5):
#    fold_accuracies = cross_val_score(classifier, inputs, y = outputs, scoring='accuracy', cv = cv, fit_params=parameters)
    folds = StratifiedKFold(n_splits=cv , shuffle = True, random_state=9)
    fold_accuracies = []
    ohe = OneHotEncoder()
    ohe.fit(outputs)
    for train, test in folds.split(inputs, outputs):
        x_train = inputs[train]
        y_train = ohe.transform(outputs[train]).toarray()
        x_test = inputs[test]
        y_test = ohe.transform(outputs[test]).toarray()
        classifier = net_generator()
        classifier.fit(x_train, 
                       y_train, 
                       epochs = parameters['epochs'], 
                       batch_size = parameters['batch_size'], 
                       verbose = 0)
        metrics = classifier.evaluate(x_test, y_test, batch_size = parameters['batch_size'], verbose = 0)
        fold_accuracies.append(metrics[1])
    return fold_accuracies