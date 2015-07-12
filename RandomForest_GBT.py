
# coding: utf-8

# In[1]:

#import Spark and MLlib packages
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, GradientBoostedTrees
from pyspark.mllib.util import MLUtils

#import data analysis packages
import numpy as np
import pandas as pd
import sklearn

from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from numpy import array

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

#for sklearn decision tree pdf plotting
from sklearn.externals.six import StringIO
import pydot

#import data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

#misc packages
from __future__ import division
from __future__ import print_function


# In[2]:

#I.Load dataset
mem = Memory("./mycache")

#using decoration to pass file to memory
@mem.cache
def get_data():
    data = load_svmlight_file("/usr/local/spark/data/mllib/sample_libsvm_data.txt")
    return data[0], data[1]

x, y = get_data()


# In[3]:

#Have to convert to dense array to fit the model
dense_x = x.toarray()

#Split the training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(dense_x, y, test_size=0.3)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[4]:

#Training the model
RFT = RandomForestClassifier(n_estimators=3, max_depth=5).fit(X_train, Y_train)
GBT = GradientBoostingClassifier(n_estimators=3, max_depth=5).fit(X_train, Y_train)


# In[5]:

def cal_model_accuracy(list):
    for i, clf in enumerate(list):
        predicted = clf.predict(X_test)
        expected = Y_test
    
        #compare results
        accuracy = metrics.accuracy_score(expected, predicted)
        if i==0: print("Random Forest accuracy is {}".format(accuracy))
        else:    print("Gradient Boosting accuracy is {}".format(accuracy))

cal_model_accuracy((RFT, GBT))


# In[6]:

#IV Use MLlib
sc = SparkContext("local", "Ensemble_Tree")


# In[7]:

data = MLUtils.loadLibSVMFile(sc, '/usr/local/spark/data/mllib/sample_libsvm_data.txt')


# In[8]:

#Split the training set and test set
(trainingData, testData) = data.randomSplit([0.7, 0.3])


# In[9]:

#Training model
RF_model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                        numTrees=3, featureSubsetStrategy="auto", 
                                        impurity='gini', maxDepth=5, maxBins=32)

GB_model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={}, numIterations=3)


# In[10]:

#Predication
def cal_mllib_accuracy(list):
    for i, clf in enumerate(list):
        #prediction with the features
        predictions = clf.predict(testData.map(lambda x: x.features))
        #append with lables first then features
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        
        accuracy = labelsAndPredictions.filter(lambda (v, p): v == p).count()/testData.count()
    
        #compare results
        
        if i==0: print("PySpark RandomForest accuracy is {}".format(accuracy))
        else:    print("PySpark GradientBoosted accuracy is {}".format(accuracy))
            
cal_mllib_accuracy((RF_model, GB_model))

