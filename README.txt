HOW TO RUN
=============================

python MP4_1.py iris.arff [optional: testSet.arff]

DOCUMENTATION
==============================

The testSet file is optional. If the the testSet is not provided, then the dataset is divided into two equal sets, training set and a test set, where the data rows are selected randomly. Therefore, the accuracy changes with each run of the program. 

I have used a Gaussian distribution to calculate the probabilities. 
Currently, my implementation only supports ARFF files that contain only numbers in the data rows and the class as the last column in the rows.
For e.g.:
			1,2,3,4,classA
			2,3,4,5,classB
			...
			...


