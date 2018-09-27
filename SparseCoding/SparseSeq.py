#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:44:33 2018

@author: mpcr
"""

#self-organizing map#


#import the libraries and utilities
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from scipy.misc import bytescale

##get data and format data
#import excel file data#
data = np.genfromtxt('DE0.csv', delimiter=',', missing_values='NA', filling_values=1, skip_header=2, usecols=range(1,7))

#view the data shape
print (data.shape)

#remove the data inputs that start with value 0
remove = data[:,0] != 0
data = data[remove,:]
removeNA = data[:, -1] != 1
data = data[removeNA, :]

##seperate data into training and testing set
#colelct test data
testnum = int(0.1 * data.shape[0])
randtestind = np.random.randint(0,data.shape[0], testnum)
testdata = data[randtestind, :]
#remove test data from all data
data = np.delete(arr=data, obj=randtestind, axis= 0)

print testdata.shape
print data.shape


# normalize the data
data -= np.mean(data, 0)
data /= np.std(data, 0)
print(np.mean(data), np.std(data))
plt.hist(data[:,0], bins=100)
plt.show()

#check data shape
print(data.shape)

##define variables/parameters for network--> build the network
#define numper of input values (columns)
n_in = data.shape[1]
#define weight matrices multiplication and number of nodes
w = np.random.randn(3, n_in) * 0.1
#learning rate
lr = 0.01
#number of iterations to update parameters/nodes
n_iters = 10000

#do the training
for i in range(n_iters):
    randsamples = np.random.randint(0, data.shape[0], 1)[0] #get randdom samples
    rand_in = data[randsamples, :] #get the random samples from the dataset
    #get nodes to look at data and asses activation
    difference = w - rand_in
    #print difference.shape
    #how to tell activation of each node
    dist = np.sum(np.absolute(difference), 1)
    #update most activated for lowest distance
    best = np.argmin(dist)
    #update weights
    w_eligible = w[best,:]
    w_eligible += (lr * (rand_in - w_eligible))
    w[best,:] = w_eligible
    cv2.namedWindow('weights', cv2.WINDOW_NORMAL) #creates window to show image in
    cv2.imshow('weights', bytescale(w)) #show this in the window
    cv2.waitKey(1) #pause every 1 ms after showing image

##validation
#run testdata through to see which node activated the most

node1w = w[0, :]
node2w = w[1, :]
node3w = w[2, :]

difference1 = node1w - testdata
dist1 = np.sum(np.absolute(difference1), 1)

difference2 = node2w - testdata
dist2 = np.sum(np.absolute(difference2), 1)

difference3 = node3w - testdata
dist3 = np.sum(np.absolute(difference3), 1)
    
#cv2.destroyAllWindows() # destroy the cv2 window
print w

    
    