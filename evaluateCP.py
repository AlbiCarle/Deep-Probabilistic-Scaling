import pandas as pd
import numpy as np
import os 

def average_CPerror(y_test, conformal_preds):
	errors_cnt=[]
	for test_row in range(len(y_test)):
		#print(y_test[test_row], conformal_preds[test_row])
		if y_test[test_row] not in conformal_preds[test_row]:
			errors_cnt.append(1)
		else:
			errors_cnt.append(0)
	return np.mean(errors_cnt),np.std(errors_cnt)

def single_class_error(y_test,label,conformal_preds):
	errors=[]
	for test_row in range(len(y_test)):
		if y_test[test_row]==label and y_test[test_row] not in conformal_preds[test_row]:
			errors.append(1)
		else:
			if y_test[test_row]!=label:
				continue
			else:
				errors.append(0)
	#n_class=y_test[y_test==label].count()
	#print(n_class)
	return np.mean(errors),np.std(errors)


def averageNumClasses(conformal_preds):
	numC=[]
	for cp_instance in conformal_preds:
		numC.append(len(cp_instance))
	return np.mean(numC),np.std(numC)

def numLabels(conformal_preds):
	numC=[]
	for cp_instance in conformal_preds:
		numC.append(len(cp_instance))
	return numC

def getEmptyCP(conformal_preds):
	numEmpty=[]
	for cp_instance in conformal_preds:
		if len(cp_instance)==0:
			numEmpty.append(1)
		else:
			numEmpty.append(0)

	return np.mean(numEmpty),np.std(numEmpty)

def getSingletonCP(conformal_preds):
	numEmpty=[]
	for cp_instance in conformal_preds:
		if len(cp_instance)==1:
			numEmpty.append(1)
		else:
			numEmpty.append(0)

	return np.mean(numEmpty),np.std(numEmpty)

def getSingletonByClass(y_test,label,conformal_preds):
	numEmpty=[]
	for cp_instance in conformal_preds:
		if label in cp_instance and len(cp_instance)==1:
			numEmpty.append(1)
		else:
			numEmpty.append(0)
	return np.mean(numEmpty),np.std(numEmpty)

def getMultipleCP(conformal_preds):
	numEmpty=[]
	for cp_instance in conformal_preds:
		if len(cp_instance)>1:
			numEmpty.append(1)
		else:
			numEmpty.append(0)

	return np.mean(numEmpty),np.std(numEmpty)



def EvaluateConformal(results,Y_test,cls0label,cls1label):
	# VALIDITY 
	# average error considering both classes
	avgErr,stdErr=average_CPerror(Y_test,results)# può anche essere utile per singole classi; forse da confrontare con tasso medio di errore senza conformal
	# average error for single classes 
	errSingleClass0,stderr0=single_class_error(Y_test,cls0label,results)
	errSingleClass1,stderr1=single_class_error(Y_test,cls1label,results)
	# EFFICIENCY
	avgC,stdC=averageNumClasses(results)
	# number of empty regions wrt all CPs
	avgEmpty,stdEmpty=getEmptyCP(results)

	avgSingle,stdSingle=getSingletonCP(results)

	avgSingle0,stdSingle0 = getSingletonByClass(Y_test,cls0label,results)
	avgSingle1,stdSingle1 = getSingletonByClass(Y_test,cls1label,results)
	
	avgDouble,stdDouble=getMultipleCP(results)

	return avgErr,stdErr, errSingleClass0, stderr0, errSingleClass1, stderr1, avgC, stdC, avgEmpty, stdEmpty, avgSingle,stdSingle, avgSingle0,stdSingle0, avgSingle1,stdSingle1,avgDouble,stdDouble




