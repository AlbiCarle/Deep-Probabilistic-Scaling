import numpy as np


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

def CNN_Model(image_shape):
    
    model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=image_shape),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid'), 
    ])
    
    return model

def CNN_Model_Imbalanced(image_shape, output_bias = None):
    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=image_shape),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid',bias_initializer=output_bias), 
    ])
    
    return model

def CNN_Model2(image_shape):
    # removed last layer wrt previous model
    model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=image_shape),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid'), 
    ])
    
    return model

def EvaluateModel(model, X_test, Y_test, showCM = True):
    
    p_ts = model.predict(X_test)-0.5
    y_pred_ts = []
    for p in p_ts:
        if p < 0:
            y_pred_ts.append(0)
        else:
            y_pred_ts.append(1)
    if showCM:
        cm_svm = confusion_matrix(Y_test, y_pred_ts)
        cmSVM = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
        cmSVM.plot()
        cmSVM.ax_.set_title("{}".format("Test set - CNN"))
        plt.show()

    TN, FP, FN, TP = confusion_matrix(Y_test, y_pred_ts).ravel()


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
    
    
    return p_ts, y_pred_ts, accuracy,f1,PPV,NPV,TPR,TNR,FPR,FNR

