# %% Imports

from sympy import Matrix, init_printing
import sympy as sym
import sympy.printing as printing
from sympy import Integral, Matrix, pi, pprint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpmath as mp
import VectorFunctions as vc

# %% All calculations relating to the vectors

# initialise vectors for test lines

r1 = np.array([2,6,-9])
r2 = np.array([-1,-2,3])

v1 = np.array([3,4,-4])
v2 = np.array([2,-6,1])

print(vc.FindShortestDistance(r1,r2,v1,v2))
print(vc.LocShortestDistance(r1,r2,v1,v2))

c1, c2, dist = vc.LocShortestDistance(r1,r2,v1,v2)

t = np.linspace(0,2000,2000)

camera_1_theta = np.radians(45)
camera_1_phi = np.radians(10)

camera_1_r0 = np.array([0,0,0])
camera_1_r0, camera_1_vector1 = vc.Polar2Vector(camera_1_r0, camera_1_theta, axis="xz", camera="1")
camera_1_r0, camera_1_vector2 = vc.Polar2Vector(camera_1_r0, camera_1_phi, axis="yz", camera="1")

camera_1_r1 = t*camera_1_vector1.T
camera_1_r2 = t*camera_1_vector2.T

x1 = -100
z1 = 100

camera_2_theta = np.radians(-30)
camera_2_phi = np.radians(10)

camera_2_r0 = np.array([x1, 0, z1]) # np.array([-mpmath.tan(camera_2_theta)*x1,0,z1])
camera_2_r0, camera_2_vector1 = vc.Polar2Vector(camera_2_r0, camera_2_theta, axis="xz", camera="2")
camera_2_r0, camera_2_vector2 = vc.Polar2Vector(camera_2_r0, camera_2_phi, axis="yz", camera="2")

camera_2_r1 = t*camera_2_vector1.T
camera_2_r2 = t*camera_2_vector2.T

# remember y and z axis are switched 

plt.figure('3D plot 2 Cameras Both Angles')
ax = plt.axes(projection='3d')

ax.plot3D(xs=camera_1_r1[0]+camera_1_r0[0],ys=camera_1_r1[2]+camera_1_r0[2],zs=np.zeros(camera_1_r1.shape[-1]))
ax.plot3D(xs=camera_2_r1[0]+camera_2_r0[0],ys=camera_2_r1[2]+camera_2_r0[2],zs=np.zeros(camera_1_r1.shape[-1]))

ax.plot3D(xs=np.zeros(camera_1_r2.shape[-1]),ys=camera_1_r2[2]+camera_1_r0[2],zs=camera_1_r2[1]+camera_1_r0[1])
ax.plot3D(xs=np.zeros(camera_2_r2.shape[-1]),ys=camera_2_r2[2]+camera_2_r0[2],zs=camera_2_r2[1]+camera_1_r0[1])

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
ax.set_xlim([-100,100])
ax.set_ylim([0,200])
ax.set_zlim([-100,100])
ax.elev = 90

plt.figure('3D plot Camera 1 Both Angles')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

ax.plot3D(xs=camera_1_r1[0]+camera_1_r0[0],ys=camera_1_r1[2]+camera_1_r0[2],zs=np.zeros(camera_1_r1.shape[-1]))
ax.plot3D(xs=np.zeros(camera_1_r2.shape[-1]),ys=camera_1_r2[2]+camera_1_r0[2],zs=camera_1_r2[1]+camera_1_r0[1])

camera_1_cart = vc.sph2cart(1, camera_1_theta, camera_1_phi)
camera_2_cart = vc.sph2cart(1, camera_2_theta, camera_2_phi)

camera_1_vector = camera_1_cart[1] - camera_1_r0
camera_2_vector = camera_2_cart[1] - camera_2_r0

camera_1_t_vector = t*np.array([camera_1_vector]).T
camera_2_t_vector = t*np.array([camera_2_vector]).T

plt.figure('3D plot Camera 1 & 2 Cart')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

ax.set_xlim([-100,100])
ax.set_ylim([-100,100])
ax.set_zlim([0,200])
ax.elev = 90


ax.plot3D(xs=camera_1_t_vector[0]+camera_1_r0[0],ys=camera_1_t_vector[2]+camera_1_r0[2],zs=camera_1_t_vector[1]+camera_1_r0[1])
ax.plot3D(xs=camera_2_t_vector[0]+camera_2_r0[0],ys=camera_2_t_vector[2]+camera_2_r0[2],zs=camera_2_t_vector[1]+camera_2_r0[1])

d = vc.FindShortestDistance(camera_1_r0, camera_2_r0, camera_1_vector, camera_2_vector)
position_short_dist = vc.LocShortestDistance(camera_1_r0, camera_2_r0, camera_1_vector, camera_2_vector)

plt.show()
# %%