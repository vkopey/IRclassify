# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# емпіричні (експертні) оцінки
x=np.array([
       [ 5,  9,  10],
       [ 5,  7,  9],
       [ 7,  2,  6],
       [ 8, -2,  4],
       [ 9, -5,  3],
       [ 10, -10,  1],
       [ 10, -10, -3],
       [ 10, -5, -3],
       [ 7, -4, -5],
       [ 4,  2, -8],
       [ 3,  6, -9],
       [ 2,  10, -10],
       [0,  10, -10],
       [-2,  5, -8],
       [-5,  2, -8],
       [-10, -3, -5],
       [-7, -7, -3],
       [-8, -10, -3],
       [-10, -10,  2],
       [-10, -8,  4],
       [-10, -3,  4],
       [-0,  2,  7],
       [-6,  6,  10],
       [-2,  10,  10]])
x=x/10.0
# теоретичні оцінки
yt=np.linspace(pi/8, 2*pi-pi/8, 8)
xt=np.array([np.sin(yt)])
yt=np.arange(8)
xt=xt.repeat(3)
xt=xt.reshape(-1,1)
x=np.hstack([x,xt]) # доповнити емпіричні дані теоретичними
# мітки класів (мультикласова класифікація)
y=np.array( [0,0,0,1,1,1,2,2,2,3,3,3, 4,4,4,5,5,5,6,6,6,7,7,7] )

model=DecisionTreeClassifier(max_depth=4)
#model=GradientBoostingClassifier(max_depth=4)
model.fit(x,y)
print model.predict(np.array([[1],[1],[1],[1]]).T)
print model.predict_proba(np.array([[1],[1],[1],[1]]).T)

s=cross_val_score(model, x, y, cv=3)
print s, s.mean()
