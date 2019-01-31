# Machine Learning Using Decision Tree

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The following code gives the idea of Supervised Machine Learning using decision Tree.
The problem statement is As Follows

  - we want to predict whether a person survived or not.
  - We have the datasets of all the passengers that we'll use to train our Model(Check Datasets Folders)

# PreProcessing The Data

  - Removing Null Values
  - Removing Unique Values(To Avoid OverFitting)
  - Encoding Data

You can also:
  - Add Additional features created using previous for more accuracy

Decision tree is the most powerful and popular tool for classification and prediction.

Construction of Decision Tree:

   - A Classifier(Tree Structure)
   - Decision Node(Test)
   - Leaf Node(Classification/Value)

### Libraries
The Python Libraries used for the problem statement are:  

* [Sklearn] - https://scikit-learn.org/
* [Numpy] - http://www.numpy.org/
* [Pandas] - https://pandas.pydata.org/
* [MatPlot] - https://matplotlib.org/
* [SeaBorn] - https://seaborn.pydata.org/introduction.html

### Model Creation
Best selection of features using crossFitting
- GridSearch in our Case(https://scikit-learn.org/stable/modules/grid_search.html)
### Model training
- Using the training datasets(see Datasets/train.csv) after preprocessing to train the model

### Prediction

After training the data, The model is set to predict the outcome(Survived passengers in our case) using the testing datasets(see Datasets/test.csv)

### Accuracy
The Predicted Data is then compared with the actual datasets(see Datasets/gender_submission.csv) to check the accuracy of our machine.
The More the accuracy, the better our prediction is.
   
### References 
https://www.kaggle.com/c/titanic

  
