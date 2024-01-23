import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import pathlib
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten,Activation, BatchNormalization,MaxPooling2D
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, f1_score, recall_score



def SplitDataset(X, Y, delta, epsilon):
    
    n_c = int(np.ceil((7.47) / epsilon * np.log(1 / delta)))

    X_learn, X_test, Y_learn,Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 5)

    X_train, X_cal, Y_train, Y_cal = train_test_split(X_learn,Y_learn, test_size = n_c, random_state = 5)
    
    return X_train, Y_train, X_cal, Y_cal, X_test, Y_test



