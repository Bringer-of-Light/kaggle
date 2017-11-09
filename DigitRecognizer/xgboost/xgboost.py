#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, ParameterGrid
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

#read input
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1]
train_images, valid_images,train_labels, valid_labels = train_test_split(images, labels, train_size=0.95,random_state=0)
train_images = images;
train_labels = labels;

#preprocess
valid_images[valid_images>0]=1
train_images[train_images>0]=1

valid_images.fillna(0,inplace=True)
train_images.fillna(0,inplace=True)

##plot
#i=1
#img=train_images.iloc[i].as_matrix().reshape((28,28))
#plt.title(train_labels.iloc[i])
##plt.hist(train_images.iloc[i])
#plt.imshow(img)

#fit model
#model = svm.SVC()
#model = KNeighborsClassifier()
#model = LogisticRegression(multi_class='ovr')
#model = RandomForestClassifier(n_estimators=2000)
#model = GradientBoostingClassifier(n_estimators=100)
model = xgb.XGBClassifier()

#搜索超参数
grid = {'max_depth': [2,4,6],'n_estimators': [100,500,1000]}
bestValidScore = 0
for g in ParameterGrid(grid):
    model.set_params(**g)
    
    tic = time.clock()
    model.fit(train_images,train_labels.values.ravel())
    toc = time.clock()
    print('fit time: {0:.0f} seconds'.format(toc-tic))

    curValidScore = model.score(valid_images,valid_labels.values.ravel())
    print('current valid score:{0}, para:{1}'.format(curValidScore,g))
    # save if best
    if curValidScore > bestValidScore:
        bestValidScore = curValidScore
        trainScore = model.score(train_images,train_labels.values.ravel())
        bestGrid = g
        #序列化model和bestGrid
        f = open('xgbBestModel.txt','wb')
        pickle.dump(model,f)
        f.close()
        f = open('xgbBestModelPara.txt','wb')
        pickle.dump(bestGrid,f)
        f.close()
        
                
print('tarin score:{0}, best valid score:{1}'.format(trainScore, bestValidScore))
print('best para:',bestGrid)

#反序列化最好的model，用于预测
f = open('xgbBestModel.txt','wb')
model = pickle.load(f)
f.close()
resultName = 'results_xgb.csv'

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
df.to_csv(resultName, header=True, index=False)
