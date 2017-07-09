#!/bin/env python3
# -*- coding: cp1251 -*-

'''
 Thomas H. Cormen
 Introduction to algorithms
 '''
'''
#1. Insertion sort (���������� ��������)

def insertion_sort(A):
    for j in range(0, len(A)):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i = i - 1
        A[i+1] = key
    print(A)

insertion_sort([5,2,1,3,2,6,8,4,9])

a = [5,2,1,3,2,6,8,4,9]
print(a)
a.sort()
print(a)

#1.1 search (������ ������)
## ����������� ���������� ������� � ������ � �������� ���
def search(v,A):
    b = []
    for i in range(0, len(A)):
        if A[i] == v:
           b.append(i+1)
            #print(j+1)
        else:
            j = - 1
    #for i in range(0,len(b)):
        #print b[i]+1
    print(list(b))
    
search(2,[2,3,1,4,2,6])
'''

'''
# ?
#2. ����� ������������ (��������)

def merge(A,p,q,r):
    n1 = q - p + 1
    n2 = r - q
    L = []
    R = []
    for i in range(1,n1):
        L[i] = A[p+i-1]
    for j in range(1,n2):
        R[j] = A[q+j]
    L[n1+1] = 0
    R[n2+1] = 0
    i = 1
    j = 1
    for k  in range(p,r):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1

def merge_sort(A,p,q):
    if p < r:
        q = (p+r)/2
        merge_sort(A,p,q)
        mergq_sort(A,q+1,r)
        merge(A,p,q,r)
    print(list(A))

A = [2,4,5,7,1,3]
r=len(A)/2
merge_sort(A,1,len(A))
'''
'''
from pylab import*
x=linspace(5.7,17,50)
x1=linspace(4.2,17,50)
plot(x,5*sin(x)/x, 'k')
#plot(x,1-x**2/6+x**4/120)
plot(x1, x1-x1-0.5, 'k')
plot([4.5, 4.5], [-10, 10], 'k')
ylim(-0.6,1.1)
xlim(4.2,17)
axis('off')
#xlabel('r')
text(17.05,-0.46,'$r$', fontsize=17)
text(4.7,0.95,'$I$', fontsize=17)
plot([7.7, 7.7], [-0.5, 0.63], 'k--')
plot([14.04, 14.04], [-0.5, 0.35], 'k--')
text(7.7,-0.6,'$r_1$', fontsize=15)
text(14.04,-0.6,'$r_2$', fontsize=15)
show()
'''
"""
import math

# !!! ����������� ���� �� ������� Matplotlib
import pylab

# !!! ����������� ����� �� ���������������� ���������
from matplotlib import mlab

# ����� �������� ������ ���� �������
def func (x):    
    if x == 0:
        return 1.0
    return math.cosh(40*x-1)/x

# �������� ��������� ���������� �� ��� X
xmin = 0.0000000001
xmax = 0.001

# ��� ����� �������
dx = 0.01

# !!! �������� ������ ��������� �� ��� X �� ������� [-xmin; xmax], ������� �����
xlist = mlab.frange (xmin, xmax, dx)

# �������� �������� ������� � �������� ������
ylist = [func (x) for x in xlist]

# !!! �������� ���������� ������
pylab.plot (xlist, ylist, 'k')

#xlabel('$\alpha$')
#ylabel('$d$')

# !!! ������� ���� � ������������ ��������
pylab.show()
"""



'''
import numpy as np
A = 1.5 # ������� �����
A = np.array([1.5]) # � ����� � ���

B = np.diag((1.32, 0.78)) # �����������, �������, ������ ������������ �������
B[0, 1] = 3.12 # ����� ���������� ������� B_12 (�����������, ��� �� �������� � ����������)

C = np.zeros((1, 2)) # ������� ������ ������-������ � ��������� ������,
C = np.concatenate((C, [[3, 2.]])) # ��������� ����� ������-������ (3, 2)
C[0][0] = 0.97 # � ��� �������� ������� ������� (1,1) ���������
C = C.T # � ������ �������������, ���� ��� ��������

D = np.array([10, 12], dtype=float) # ������ ������ ������-������� � �������� �������,
# � ����� ������ ���� (�� � ��� dtype ���������� ����� �������, ��� ��� ������� ����� ��� ����)

E = np.ones((2, 1), dtype=float) # ������-������� �� ������
E *= 2.3 # �������� ��� ��������

M = A * B * C * D + E # ����� ������� ����!!!

print B
print C
print D
print E
print M
'''

'''
print( np.linalg.eig(np.matrix('-1 -6; 2 6')) )

 �������:

[[-1 -6]  # ���� �������
 [ 2  6]]
 
(array([ 2.,  3.]), matrix([[-0.89442719,  0.83205029],
        [ 0.4472136 , -0.5547002 ]])) # ���������

��� [ 2., 3.] -- � ����������� ��������, �
	
-0.89442719
0.4472136

�

	
0.83205029
 -0.5547002

-- ��� ����������� �������.
'''

#import numpy as np

#x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

#type(x)
#  x.shape # �����/��������� �������: 2 ������ � 3 �������
# x.dtype # ��� ������

#print x[1,2]

'''
y = x[:,1] # ������ --- slicing
y[0] = 9
print x

z = x[1,:]
z[2]=7
print x

a = np.ndarray(shape=(2,2), dtype=float, order='F')
b =  np.ndarray(shape=(2,2), dtype=float, order='C')

print a, b
'''

'''
# ������������� ������� 

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

def f(x):
    return math.log(x)

def p1(): 
    x = 1.0
    while x < 10.0:
        print x, "\t", f(x)
        x += 1.0

def main(): 
    p1() 
    return 0 

if __name__ == '__main__':
    main()


# ��������������

import numpy as np
from scipy.integrate import quad

def f(x): # ��� �� ���������� ������� f(x)=x**2
    #return x**3 + np.tan(x)
    #return (x-1)/np.log(x) # ��������� ���. 388
    #return  np.exp(-x**2) # ��������� ���. 389
    return x*np.exp(x)

#print quad(f, 0, 0.5)
#print np.log(2)
#print quad(f,0,1)


# ���������������� ���������
# ������� odeint(f(x,y),x0,<������ x>)

from pylab import *
from scipy.integrate import *


t = linspace(0,6*pi,10000) # ����������� ��������� �������

def f(v,t):
    return -(2/pi)*arctan(1000*v)+1.2*cos(t)

v = odeint(f,0,t)

plot(t,v,lw=2)
xlabel('t')
ylabel('v')
xlim(0,6*pi)
grid()
show()
#savefig('du1.png')

# ��� 1 �������
x = linspace(1,2*np.pi,100)

def f(x,y):
    return np.sin(x) + np.exp(-x)*np.sin(2*x) + np.exp(-x**2)*np.sin(3*x) + np.exp(-x**3)*np.sin(4*x)
    
y = odeint(f,1,x)

plot(x,y)
xlabel('x')
ylabel('y')
grid()
show()

'''
"""
# ��� 2 �������

import numpy as np
from pylab import *
from scipy.integrate import *

t = linspace(0,40*np.pi,10000)

def f(y,t):
    x,v = y
    return [v, -(1-x**2/100)*x + np.cos(t)]

result = odeint(f,[0,0],t)
x = result[:,0]
v = result[:,1]# ������ �� ������ ������

plot(t,x)
xlabel('t')
ylabel('x')
grid() # �����
show()

# ������� �������

rc('figure',figsize=(6.25,6))
figure()
plot(x,v,lw=2)
xlabel('x')
ylabel('v')
#xlim(-10,10)
#ylim(-10,10)
grid()
#savefig(�example10a.pdf�)
show()
"""

'''
# ��� 3 �������

import numpy as np
from pylab import *
from scipy.integrate import *

x = linspace(0,40,100)

def f(t,x):
    y, v, z = t
    return [v, z, np.sin(x)]

result = odeint(f,[1,0.6,-1],x) # ��� �� [1,0.6,-1] ����� y = cos(x) + 0.6*x
y = result[:,0]
v = result[:,1]# ������ �� ������ ������
z = result[:,2]

plot(x,y)
xlabel('x')
ylabel('y')
grid() # �����
text(5,20,'$y \'\'\'=\sin (x)$', fontsize=15)
text(5,18,'$y = \cos (x) + 0.6x$', fontsize=15)
show()
'''


# �������� �������. ����

import numpy as np
from scipy.linalg import solve
'''
# ������ ������� �������� ���������
# Solve the equation a x = b for x.

print solve(np.array([[2, 1, -2], [1, -2, 3], [7, 1, -1]]), np.array([5, -3, 10]).T)


# ������������

def det(a):
    A = a.copy()
    d = 1
    for i in range(len(A)-1):
        if A[i, i] == 0:
            d = -d
            for j in range(len(A)):
                A[i,j], A[i+1,j] = A[i+1,j], A[i,j]
        for j in range(i + 1, len(A)):
            A[j] = A[j] - A[i] * A[j, i] / A[i, i]
    print(A)
    for i in range(len(A)):
        d *= A[i, i]
    return d


A = np.array([[0,2,3], [4,5,6], [7,8,9]], dtype=float)
print(A)
print(det(A))
print(np.linalg.det(A))


# ����������� �����

a0 = complex(2, 5)
b0 = complex(1,-7)

a1 = complex(1,2)
b1 = complex(3,-1)

a2 = complex(23,1)
b2 = complex(3,1)

a3= complex(2,5)

a4 = complex(21,-20)

#print a0+b0, ',', a1*b1, ',', a2/b2, ',', a3**3, ',', np.sqrt(a4)

a5 = complex.conjugate(a0) # ����������

a6 = pow(a0, b0)

#print a5, ',', (3-4j).conjugate(), ',', a6, ',', pow(2,3)

print pow(complex(1,1),complex(1,1))
'''

#print abs(3+4j)
# exit('error') # ������ ���������� ���������� ��������� � �������������� ����������

'''
import time
print time.gmtime(0)


import math, cmath
print cmath.sqrt(-1)
print math.sqrt(-1)


import random as r
print int(r.uniform(1,1000))


import numpy as np
a0 = np.zeros((2, 3), 'float')
print 'a0:.\n', a0
a1 = np.ones((3, 2), 'int')
print 'a1:\n', a1
a2 = np.eye(3, 2, dtype=int)
print 'a2:\n', a2
a3 = np.identity(4, 'float')
print 'a3:\n', a3


# C���
import numpy as np
a = np.mat([[5, 1], [2, 3]])
print 'a:\n', a
c = np.mat( [1, 1]) .T
print 'c:\n', c
x = np.linalg.solve(a, c)
print 'solution x:\n', x


import numpy as np
p = np.poly1d([1, -4, 3])
print np.poly1d(p)
print p(0.5), np.polyval(p, 0.5) # �������� �������� � �����
print 'roots:\n', np.roots(p)


# ����������� �������� �������, ������� ������ � ������������
import numpy as np
import matplotlib.pyplot as pit
x = np.linspace(0., 1., 100)
y = np.exp(-x)
e = 0.1*np.random.normal(0, 1, 100)
pit.errorbar(x, y, e, fmt='.')
pit.show()



# �����-�������
import scipy.special as spec
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.01)
y = spec.gamma(x)
for i in x:
    if y[i] > 1.e20: y[i] = 1.e20
    if y[i] < -1.e20: y[i] = -1.e20
plt.plot(x, y, label=r'$\Gamma(x)$')
plt.axis([-5.,5.,-20.,20.])
plt .xlabel('x')
plt.legend(loc=4)
plt.grid(True)
plt.show()


# ����� ��������

import numpy as np
import math as mt
def bisection(f, x1, x2, tol=1.0e-10):
    f1 = f(x1)
    f2 = f(x2)
    if f1*f2 > 0.:
        print 'f(xl) and f(x2) can not have the same signs'
    n = int(mt.ceil(mt.log(abs(x2 - x1)/tol)/mt.log(2.)))
    for i in range(n):
        x3 = 0.5*(x1 + x2)
        f3 = f(x3)
        if f2*f3 < 0.:
            x1 = x3
            f1 = f3
        else:
            x2 =x3
            f2 = f3
    return (x1 + x2)/2.


import matplotlib.pyplot as pit
# ��������� ���� f(x)=0
def f(x):
    return (1. + x**2) * np.exp(-x) + np.sin(x)
    #return 4.*np.sin(x)+1-x
    #return x**2-10.*(np.sin(x))**5
x = np.linspace(-1., 10., 200)
y = f(x)
pit.plot(x, y, 'b')
"""
x1 = np.linspace(0., 10., 200)
y1 = f(x1)*0
pit.plot(x1, y1, 'b')
"""
pit.xlabel('x')
pit.grid(True)
pit. show()
xRoot1= bisection(f, 3.4, 3.6)
print 'root(1) = ', xRoot1
xRoot2 = bisection(f, 6.03, 6.2)
print 'root(2) = ', xRoot2
xRoot3 = bisection(f, 9.3, 9.5)
print 'root(3) = ', xRoot3

# ���������� ���������� ������� ������� �������� 
def vialSort(A):
    for j in range(len(A)-1):
        for i in range(len(A)-1):
            if A[i+1] < A[i]:
                A[i], A[i+1] = A[i+1], A[i]
    print A

a=[5,4,2,1,6,3]
vialSort(a)
'''














