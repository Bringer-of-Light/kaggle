#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#read input
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1]
#train_images, valid_images,train_labels, valid_labels = train_test_split(images, labels, train_size=0.8,test_size=0.2, random_state=0)
train_images = images;
train_labels = labels;

#preprocess
#valid_images[valid_images>0]=1
train_images[train_images>0]=1

#valid_images.fillna(0,inplace=True)
train_images.fillna(0,inplace=True)

#plot
i=1
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.title(train_labels.iloc[i])
#plt.hist(train_images.iloc[i])
plt.imshow(img)

#fit model
#model = svm.SVC()
#model = KNeighborsClassifier()
#model = LogisticRegression(multi_class='ovr')
model = RandomForestClassifier(n_estimators=2000)

model.fit(train_images, train_labels.values.ravel())
print(model.score(train_images,train_labels), model.score(valid_images,valid_labels))

#make predict
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
test_data.fillna(0,inplace=True)
results=model.predict(test_data)

#save results
df = pd.DataFrame(results)
df.reset_index(inplace=True)
df.columns=['ImageId','Label']
df['ImageId']+=1
df.to_csv('results.csv', header=True, index=False)
