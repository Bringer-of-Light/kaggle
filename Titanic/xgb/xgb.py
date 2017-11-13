#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from copy import deepcopy

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
images = labeled_images.iloc[:100,1:]
labels = labeled_images.iloc[:100,:1]
images[images>0]=1
images.fillna(0,inplace=True)
train_images, valid_images,train_labels, valid_labels = train_test_split(images, labels, train_size=0.95,random_state=0)

#preprocess

##plot
#i=1
#img=train_images.iloc[i].as_matrix().reshape((28,28))
#plt.title(train_labels.iloc[i])
##plt.hist(train_images.iloc[i])
#plt.imshow(img)

#fit model
model = xgb.XGBClassifier()

#搜索超参数
grid = {'max_depth': [6,8],'n_estimators': [1000,1500]}
bestValidScore = 0
trainScore = 0
bestModel = 0
bestPara = 0
for para in ParameterGrid(grid):
    model.set_params(**para)
    
    tic = time.clock()
    model.fit(train_images,train_labels.values.ravel())
    toc = time.clock()
    print('fit time: {0:.0f} seconds'.format(toc-tic))

    curValidScore = model.score(valid_images,valid_labels.values.ravel())
    curTrainScore = model.score(train_images,train_labels.values.ravel())
    print('current valid score:{0},\n'
          'current train score,{1},\n'
          'para:{2}\n'.format(curValidScore,curTrainScore,para))
    # save if best
    if curValidScore > bestValidScore:
        bestValidScore = curValidScore
        trainScore = curTrainScore
        bestPara = para
        bestModel = deepcopy(model)
        # 序列化model和bestPara
        f = open('xgbBestModel.txt', 'wb')
        pickle.dump(bestModel, f)
        f.close()
        f = open('xgbBestModelPara.txt', 'wb')
        pickle.dump(bestPara, f)
        f.close()

#reset model to bestModel
model = bestModel
       
print('tarin score:{0}, best valid score:{1}'.format(trainScore, bestValidScore))
print('best para:',bestPara)

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

resultName = 'results_xgb.csv'
df.to_csv(resultName, header=True, index=False)
