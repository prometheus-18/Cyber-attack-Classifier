# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 17:57:12 2022

@author: user
"""
# import libraries
import numpy as np
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
import os
csv_files = []
for dirname, _, filenames in os.walk('C:/Users/user/Downloads/Attacks/MachineLearningCSV/MachineLearningCVE'):
    for filename in filenames:
        csv_file = os.path.join(dirname, filename)
        print(os.path.join(dirname, filename))
        csv_files.append(csv_file)

dataset = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# about dataset
a = dataset.dtypes
print(a)
print(dataset.head())
print(dataset.describe())


# replace infinity
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

# replace nan
dataset.fillna(0, inplace = True)

# managing labels
# np.unique(dataset.iloc[:,-1], return_counts = True)

df_experiment = dataset.copy()
df_experiment.replace("Web.*", "Web Attack", regex=True, inplace=True)
df_experiment.replace(r'.*Patator$', "Brute Force", regex=True,inplace=True)
df_experiment.replace(["DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"], "DoS", inplace=True)
df_experiment.value_counts()

print("after replacing : ",np.unique(df_experiment.iloc[:,-1], return_counts = True))


class_attack = ['PortScan', 'Web Attack', 'Brute Force', 'DDoS', 'Bot','Infiltration', 'DoS', 'Heartbleed']
df_experiment.replace(class_attack, "attack", regex=True, inplace=True)
print(np.unique(df_experiment.iloc[:,-1], return_counts = True))


# allocating values
X = df_experiment.iloc[:,:-1].values 
y = df_experiment.iloc[:,-1].values


# Encoding categorical dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print("before oversampling : ",np.unique(y,return_counts=True))


# normalize
n_X = normalize(X)
sc = StandardScaler()
s_X = sc.fit_transform(n_X)

# oversampling
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
print("starting to oversample")
X_, y_ = oversample.fit_resample(s_X, y)
print("after oversampling : ",np.unique(y_,return_counts=True))

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)

print("test train data splitted")

#Dimentioanlity reduction

print("starting dimensionality reduction")
from sklearn.decomposition import PCA
pca = PCA() 
x_train = pca.fit_transform(X_train)
x_test = pca.transform(X_test)
total=sum(pca.explained_variance_)
k=0
current_variance=0
while current_variance/total < 0.98:
    current_variance += pca.explained_variance_[k]
    k=k+1
print("number of features saved : ",k)


#Apply PCA with n_componenets

pca = PCA(n_components=k)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Exoplanet Dataset Explained Variance')  
plt.show() 


#training xgbooster model
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix")
print(cm)

# accuracy
print("accuracy = ",accuracy_score(y_test, y_pred))

#plotting cm matrix
import seaborn as sns

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# save the model
import joblib
joblib.dump(classifier,"Cyber attack classifier.pkl")


count = 0
malacious_packet = []
for i in range (len(x_test)):
    if classifier.predict(x_test[[i]]) == [1]:
        malacious_packet.append(x_test[[i]])
        count = count + 1
        if count == 10000:
            malacious_packet.clear()
            malacious_packet = []
            
print("attakcs prevented : ", count)
print("legit packets : ", len(x_test)-count)


        