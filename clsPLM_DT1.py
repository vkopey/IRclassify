# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# ознаки класів
x=np.array([[1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,0,0,0,0,0,0,1,1,1, 1,1,1,0,0,0,0,0,0,1,1,1],
            [1,1,1,1,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,1,1,1,1,1,1]])
# мітки класів (мультикласова класифікація)
y=np.array( [0,0,0,1,1,1,2,2,2,3,3,3, 4,4,4,5,5,5,6,6,6,7,7,7] )
x=x.T
model=DecisionTreeClassifier(max_depth=4)
model.fit(x,y)
print model.predict(np.array([[1],[1],[1]]).T)
print model.predict_proba(np.array([[1],[1],[1]]).T)

s=cross_val_score(model, x, y, cv=3)
print s, s.mean()