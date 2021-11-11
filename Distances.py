# %% Imports

from sympy import Matrix, init_printing
import sympy as sym
import sympy.printing as printing
from sympy import Integral, Matrix, pi, pprint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpmath


A = 50
B = 50

A, B = np.array(A), np.array(B)




# %% Find the smallest distance between two line segments

def EqOfLine(r1,r1_2):
    vector = r1 - r1_2
    #print("The vector equation of your line is: ", printing.latex('r = '), )
    return vector

def FindShortestDistance(r1, r2, v1, v2):
    n = np.cross(v1,v2)

    if n.all == 0:
        d = np.sqrt((np.cross((r2-r1),v1))**2/(np.linalg.norm(v2,axis=0))**2)
        print("The two lines are parallel")
    else:
        d = (np.dot(n,(r1-r2)))/(np.linalg.norm(n,axis=0))
        print("The two lines are not parallel")
    return d

def LocShortestDistance(r1, r2, v1, v2):
    n = np.cross(v1,v2)

    n1 = np.cross(v1, n)
    n2 = np.cross(v2, n)

    c1 = r1 + v1*(np.dot((r2-r1),n2)/(np.dot(v1,n2)))
    c2 = r2 + v2*(np.dot((r1-r2),n1)/(np.dot(v2,n1)))

    dist = np.linalg.norm(c1-c2,axis=0)

    return c1, c2, dist

def Polar2Cart(r0, theta):
    vector = np.array([[mpmath.cot(np.radians(theta)),1]])
    return r0, vector

def Polar2Cart3D(v1, v2, theta, phi):
    x = v1[0]
    y = v2[0]

    vector = np.array[x,y,1]

def Camera1toCamera2(z):
    z = np.tan()

r1 = np.array([2,6,-9])
r2 = np.array([-1,-2,3])

v1 = np.array([3,4,-4])
v2 = np.array([2,-6,1])

print(FindShortestDistance(r1,r2,v1,v2))
print(LocShortestDistance(r1,r2,v1,v2))

c1, c2, dist = LocShortestDistance(r1,r2,v1,v2)

t = np.linspace(0,1,100)
camera_1_r0 = np.array([0,0,0])
camera_1_theta = 34
camera_1_r0, camera_1_vector1 = Polar2Cart(camera_1_r0, camera_1_theta)

camera_1_phi = 0
camera_1_r0, camera_1_vector2 = Polar2Cart(camera_1_r0, camera_1_phi)

camera_1_r1 = t*camera_1_vector1.T
camera_1_r2 = t*camera_1_vector2.T

camera_2_r0 = np.array([0,0,0])
camera_2_theta = 34
camera_2_r0, camera_2_vector1 = Polar2Cart(camera_2_r0, camera_2_theta)

camera_2_phi = 0
camera_2_r0, camera_2_vector2 = Polar2Cart(camera_2_r0, camera_2_phi)

camera_2_r1 = t*camera_2_vector2.T
camera_2_r2 = t*camera_2_vector2.T


# remember y and z axis are switched 

plt.figure('3D plot')
ax = plt.axes(projection='3d')
ax.plot3D(xs=camera_1_r1[0],ys=camera_1_r1[1],zs=np.zeros(r1.shape[-1]))
ax.plot3D(xs=np.zeros(camera_1_r1.shape[-1]),ys=camera_1_r1[1],zs=camera_1_r1[0])
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
plt.show()
# %%
