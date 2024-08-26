# -*- coding: utf-8 -*-
import numpy as np
pi=np.pi
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# експертні оцінки
y=np.array([0,   1,   2,  3,   4,   5,   6,   7])
x=np.array([[0.5, 0.8, 1.0, 0.4, 0,   -1,  -1,  0]])
plt.scatter((y+pi/8)*0.802, x, c='k')

# теоретичні оцінки
#y=np.linspace(pi/8, 2*pi-pi/8, 8)
#x=np.array([np.sin(y)])
#y=np.arange(0,8)

x=x.T
model=DecisionTreeClassifier(max_depth=4)
#model=DecisionTreeRegressor(max_depth=4)

model.fit(x,y)
print model.predict(np.array([[1]]).T)
print model.predict_proba(np.array([[1]]).T)


X=np.linspace(-1, 1, 32)
#Y=model.predict(np.array([X]).T)
#plt.plot(Y,X,"o-")
X_=[]
Y_=[]
for x in X:
    p=model.predict_proba(np.array([[x]]).T)
    p=p[0]
    p=np.array(p*100,dtype=int)
    print x,p
    for i,j in enumerate(p):
        X_+=[x]*j
        Y_+=[i]*j

X_=np.array(X_)
Y_=np.array(Y_)
i=Y_.argsort()
#plt.plot(Y_[i],X_[i], "o-")
plt.plot(np.linspace(0, 2*pi), np.sin(np.linspace(0, 2*pi)),'k')
plt.plot([0,pi,pi,2*pi],[1,1,-1,-1], "r")
plt.scatter((Y_+pi/8)*0.802, X_,marker='|')
plt.xlabel('y'); plt.ylabel('x')
plt.grid()
plt.show()