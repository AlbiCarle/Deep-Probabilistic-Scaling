import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import pathlib
import random
import seaborn

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

from utilsData import *
from utilsModel import *


def ComputeRhoBar(model, X_cal, Y_cal, epsilon):
    n_c = len(Y_cal.squeeze())
    p_cal = model.predict(X_cal[Y_cal.squeeze() == 1])
    p_cal01 =  model.predict(X_cal)
    
    rho_bar = 1/2-p_cal
    rho_bar = rho_bar.squeeze()

    rho_bar01 = 1/2 - p_cal01
    rho_bar01 = rho_bar01.squeeze()
    
    qhat = np.quantile(rho_bar, np.ceil((n_c+1)*(1-epsilon))/n_c)
    
    return rho_bar, rho_bar01, qhat

def ComputeRhoStar(rho_bar, epsilon, n_c):
    rho_bar = rho_bar.squeeze()
    idx = rho_bar.argsort()[::-1]
    rho_bar_sorted = rho_bar[idx]

    r = np.ceil(epsilon*n_c/2)
    rho_star = rho_bar_sorted[int(r)]
    #print('rho_star = ',rho_star)
    return rho_star, rho_bar_sorted

def EvaluateScalingModel(model, rho_star,p_ts, X_test, Y_test, showCM = True):
    
    y_pred_ts_eps = []

    p_ts_star = p_ts + rho_star

    for p in p_ts_star:
        if p<0:
            y_pred_ts_eps.append(0)
        else:
            y_pred_ts_eps.append(1)

    plt.figure(figsize=(12, 8))

    cm_svm = confusion_matrix(Y_test, y_pred_ts_eps)
    cmSVM = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
    cmSVM.plot()
    cmSVM.ax_.set_title("{}".format("Scalable CNN"))
    plt.tight_layout()

    TN, FP, FN, TP = confusion_matrix(Y_test, y_pred_ts_eps).ravel()


    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = (2*TP)/(2*TP+FP+FN)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TP)
    print("ACC = {}, F1 = {}, PPV = {}, NPV = {}, TPR = {}, TNR = {}\n".format(accuracy,f1,PPV,NPV,TPR,TNR))

    print(f"FPR = {FPR}, FNR = {FNR}")

    print("TP = {}, FP = {}, TN = {}, FN = {}".format(TP,FP,TN,FN))

    return p_ts_star, y_pred_ts_eps, accuracy,f1,PPV,NPV,TPR,TNR,FPR,FNR

