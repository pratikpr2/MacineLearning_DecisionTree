# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:12:45 2019

@author: pratprak
"""

import pandas as pd
import numpy as np
import re
from sklearn import tree

#reading
train=pd.read_csv("...\train.csv")
test=pd.read_csv("...\test.csv")
y_true = pd.read_csv("...\gender_submission.csv")


#store our Passengers Ids for easy access
PassengerId = test['PassengerId']

original_train = train.copy()

full_data = [train, test]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] =  train["Cabin"].apply(lambda x:0 if type(x) == float else 1)
train['Has_Cabin'] = train["Cabin"].apply(lambda x:0 if type(x) == float else 1)

# Create a new feature Family Size
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
# Create A new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1 , 'IsAlone'] = 1

# Remove all Nulls in the Embarked Column
for dataset in full_data:
    dataset['Embarked']  = dataset['Embarked'].fillna('S')
    
# Remove all Nulls in Fare Column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
# Remove all Nulls in the Age  Column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)
    
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
# Define functions to extract title from the passengers
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    
    if title_search:
        return title_search.group(1)
    
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

# Grouping all title
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    
for dataset in full_data:
    #Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0 , 'male' : 1}).astype(int)
    
    #mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    #Mapping Embarked 
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    #Mapping fare
    dataset.loc[dataset['Fare'] <= 7.91]
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    #Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] ;
    
# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test = test.drop(drop_elements, axis = 1)


print(train.head(3))

#Finding correlation between features
import matplotlib.pyplot as plt
import seaborn as sns
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y = 1.05, size = 15)
sns.heatmap(train.astype(float).corr(), linewidths= 0.1, vmax=1.0, square = True, cmap=colormap,linecolor='white',annot = True)


y_train = train['Survived'].values
#Dropping the Title column from the Correlation
x_train = train.drop(['Survived', 'Title'], axis=1).values  
x_test = test.values
#Merging the training and test data
x_trainExtended = pd.concat([test,y_true], axis = 1)
print("Total data:")
print(x_trainExtended.head())
y_trainExtended = x_trainExtended['Survived'].values
x_trainExtended = x_trainExtended.drop(['Survived','Title'],axis=1).values


decision_tree = tree.DecisionTreeClassifier()

#Grid search Accuracy
param_grid = {
        'max_depth': [3,4,5,6,7,8,9],
        'min_samples_leaf': [1,2,3,4,5,6,7,8],
        'criterion': ['gini','entropy']
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = decision_tree , param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid = grid_search.fit(x_trainExtended, y_trainExtended)
best_parameters = grid.best_params_
print(best_parameters)

#checking the accuracy
from sklearn.metrics import accuracy_score
y_accu = y_true['Survived'].values
print(y_accu)
y_pred = grid_search.predict(x_test)
print(y_pred)
print("Accuracy:")
print(accuracy_score(y_trainExtended, y_pred) * 100, "%")



