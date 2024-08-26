import numpy as np
#from scipy.optimize import curve_fit
pi=np.pi
arcsin=np.arcsin
sin=np.sin
import matplotlib.pyplot as plt
pi=np.pi
fig, ax = plt.subplots()
ax.set_xlim(0,2*pi)
ax.set_ylim(-1,1)
ax.set_xticks((0, pi/2, pi, 3*pi/2, 2*pi))

class Rabc:
    P_name=""
    N_name=""
    def f(self,x):
        pass
    def plot_f(self):
        x=np.linspace(0,2*pi,100)
        y=self.f(x)
        ax.plot(x,y,'-')
    def fi(self,y):
        pass
    def plot_fi(self,y):
        x=self.fi(y)
        ax.scatter(x, np.full(x.size, y))
    def r2(self, x, y):
        return np.corrcoef(y, self.f(x))[0,1]**2 # R^2

class R1(Rabc):
    P_name="проц"
    N_name="рез"
    def f(self,x):
        return sin(4*x)
    def fi(self,y):
        x = np.array([arcsin(y)/4, -arcsin(y)/4 + pi/4])
        x = np.hstack((x, x+pi/2, x+pi, x+3*pi/2))
        if x[0]<0: x[0]=x[0]+2*pi
        return x

class R2(Rabc):
    P_name="вес"
    N_name="серй"
    def f(self,x):
        return sin(x)
    def fi(self,y):
        x = np.array([arcsin(y),-arcsin(y) + pi])
        if x[0]<0: x[0]=x[0]+2*pi
        return np.hstack((x, x, x, x)) # 4

class R3(Rabc):
    P_name="дем"
    N_name="ари"
    def f(self,x):
        return sin(2*x)
    def fi(self,y):
        x = np.array([arcsin(y)/2, -arcsin(y)/2 + pi/2])
        x = np.hstack((x, x+pi))
        if x[0]<0: x[0]=x[0]+2*pi
        return np.hstack((x, x)) # double

X=np.array([])
R=[R1(), R2(), R3()]#,R1(), R2(), R3()]
Y=[0.5, -0.9, 0.6]#, -0.7,-0.7,-0.7] # зробити багато оцінок: [[0.5], [0.9], [0.6]]
for r,y in zip(R,Y):
    r.plot_f()
    r.plot_fi(y)
    X=np.hstack((X, r.fi(y)))

m,x= np.histogram(X,bins=8, range=(0,2*pi))
p=m*1.0/m.sum() # імовірності
assert p.sum()==1
print(p)

ax.grid()
ax.set_xlabel('y'); ax.set_ylabel('x')
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(0,2*pi)
ax.set_xticks((0, pi/2, pi, 3*pi/2, 2*pi))
plt.bar(x[0:-1], p, align='edge', width=pi/4, edgecolor=(0,0,0))
ax.set_xlabel('y'); ax.set_ylabel('p')
plt.show()

##
# пошук найкращої моделі експертних даних
x=np.linspace(pi/8, 2*pi-pi/8, 8)
y=np.array([[0.5, 0.8, 1.0, 0.4, 0,   -1,  -1,  0]])
best=np.array([r.r2(x,y) for r in [R1(), R2(), R3()]]).argmax()
print(R[best])
