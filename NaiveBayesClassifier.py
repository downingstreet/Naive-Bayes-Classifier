import sys
import pprint
import random
import math
from collections import defaultdict

"""
References:

http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
https://en.wikipedia.org/wiki/Naive_Bayes_classifier
http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification
http://ebiquity.umbc.edu/blogger/2010/12/07/naive-bayes-classifier-in-50-lines/
http://weka.wikispaces.com/ARFF+%28stable+version%29

"""

def toFloat(letters):
	return [float(y) for x in letters for y in x]


def readARFF(filename):
	classes = list()
	attributes = list()
	dataset = list()
	with open(filename) as f:
		for line in f:
			line = line.replace(',', ' ').replace('{','').replace('}','')
			line = line.lower().split()
			if len(line) != 0 and line[0] != "%":
				if line[0] == "@attribute":
					if line[1] == "class":
						for i in line[2:]:
							classes.append(i)
					else:
						attributes.append(line[1])

				elif line[0] == "@relation":
					#global name_of_dataset
					name_of_dataset = line[1]
					continue
				elif line[0] == "@data":
					continue
				else:
					dataset.append(line)
	for row in dataset:
		for i, x in enumerate(row):
			try:
				row[i] = float(x)
			except ValueError:
				pass
	return [classes, attributes, dataset, name_of_dataset]


def divideDataset(dataset, splitRatio):
	trainSize = int(len(dataset) *splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


def divideByClass(dataset):
	divided = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if vector[-1] not in divided:
			divided[vector[-1]] = []
		divided[vector[-1]].append(vector[:len(vector)-1])
	return divided


def calcMean(numbers):
	return sum(numbers)/float(len(numbers))


def calcStdDev(numbers):
	avg = calcMean(numbers)
	variance = sum([pow(x-avg, 2) for x in numbers])/ float(len(numbers)) #float(len(numbers)-1
	return math.sqrt(variance)


def bunchTogether(dataset):
	bunch = [(calcMean(attribute), calcStdDev(attribute)) for attribute in zip(*dataset)]
	return bunch


def summarizeByClass(dataset):
	divided = divideByClass(dataset)
	bunch = {}
	for classVal, instances in divided.iteritems():
		bunch[classVal] = bunchTogether(instances)
	return bunch



def calcProb(x, calcMean, calcStdDev):
	dem = (2*math.pow(calcStdDev,2))											
	if dem == 0 or calcStdDev == 0:
		return 0
	exponent = math.exp(-(math.pow(x-calcMean, 2)/dem))
	return (1/(math.sqrt(2*math.pi)*calcStdDev))*exponent


def calcClassProb(bunch, inputVector):
	probabilities = {}
	for classValue, classSummaries in bunch.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			calcMean, calcStdDev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calcProb(x, calcMean, calcStdDev)
	return probabilities


def predict(bunch, inputVector):
	probabilities = calcClassProb(bunch, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestLabel = classValue
			bestProb = probability
	return bestLabel


def calcPredictions(bunch, testSet):
	predictions = []
	for  i in range(len(testSet)):
		result = predict(bunch, testSet[i])
		predictions.append(result)
	return predictions


def calcAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def confusionMatrix(testSet, predictions, classes):
	matrix = defaultdict(lambda: defaultdict(int))
	for row in range(len(testSet)):
		matrix[testSet[row][-1]][predictions[row]] += 1
	return matrix


def pretty(d, indent=0):
   for key, value in d.iteritems():
      print '\t' * indent + str(key)
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print '\t' * (indent+1) + str(value)


def main():
	if len(sys.argv) < 2:
		sys.exit("Usage: python MP4_1.py iris.arff [optional: testSet.arff]")

	filename = sys.argv[1]
	classes, attributes, dataset, name_of_dataset = readARFF(filename)
	
	if len(sys.argv) < 3:
		trainingSet, testSet = divideDataset(dataset, 0.5)
	else:
		trainingSet = dataset
		dummyA, dummyB, testSet, dummyC = readARFF(sys.argv[2])
	
	"""
	pprint.pprint(dataset)
	"""
	
	bunch = summarizeByClass(trainingSet)
	predictions = calcPredictions(bunch, testSet)
	accuracy = calcAccuracy(testSet, predictions)
	matrix = confusionMatrix(testSet, predictions, classes)

	print("Confusion Matrix: ")
	print
	pretty(matrix)
	print
	print("Accuracy : {0}".format(accuracy))
	print
	print("Error: {0}".format(100 - accuracy))
	print


if __name__ == "__main__":
	main()









			
			
		