# Deep Probabilistic Scaling pipeline

**1) Data download**
According to the specific dataset formatting, we have to generate matrices X (numpy ndarray) with the images, and Y with the labels. Save these matrices to a file for later usage.

**2) Dataset split and CNN learning**
Properly split the data to obtain a sufficienty large calibration set, a proper training set and a test set.
Build a suitable CNN and train it on the proper train set
Save the model and the datasets to files.


**3) Probabilistic Scaling**
Use the pre-trained Neural Network and apply the PS algorithm
Evaluate the performance of the scaled CNN