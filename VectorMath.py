# %% Imports
%matplotlib widget
from numpy.lib.function_base import angle
from skimage import color
from sympy import Matrix, init_printing
import sympy as sym
from sympy.core.numbers import I
import sympy.printing as printing
from sympy import Integral, Matrix, pi, pprint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import VectorFunctions as vf
import importlib
importlib.reload(vf)
import functions as f

# %%
angles = f.import_bz2('A_angles_test')

video_camera_1 = f.import_bz2('video_camera_1')
video_camera_2 = f.import_bz2('video_camera_2')

video = f.video_read('IMG_3005.mp4')

# %% All calculations relating to the vectors


# initialise vectors for test lines

r1 = np.array([2,6,-9])
r2 = np.array([-1,-2,3])

v1 = np.array([3,4,-4])
v2 = np.array([2,-6,1])

print(vf.FindShortestDistance(r1,r2,v1,v2))
print(vf.LocShortestDistance(r1,r2,v1,v2))

c1, c2, dist = vf.LocShortestDistance(r1,r2,v1,v2)

t = np.linspace(0,2000,2000)

camera_1_theta = np.radians(45)
camera_1_phi = np.radians(10)

camera_1_r0 = np.array([0,0,0])
camera_1_r0, camera_1_vector1 = vf.Polar2Vector(camera_1_r0, camera_1_theta, axis="xz", camera="1")
camera_1_r0, camera_1_vector2 = vf.Polar2Vector(camera_1_r0, camera_1_phi, axis="yz", camera="1")

camera_1_r1 = t*camera_1_vector1.T
camera_1_r2 = t*camera_1_vector2.T

x1 = -100
z1 = 100

camera_2_theta = np.radians(-30)
camera_2_phi = np.radians(10)

camera_2_r0 = np.array([x1, 0, z1]) # np.array([-mpmath.tan(camera_2_theta)*x1,0,z1])
camera_2_r0, camera_2_vector1 = vf.Polar2Vector(camera_2_r0, camera_2_theta, axis="xz", camera="2")
camera_2_r0, camera_2_vector2 = vf.Polar2Vector(camera_2_r0, camera_2_phi, axis="yz", camera="2")

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

camera_1_cart = vf.sph2cart(1, camera_1_theta, camera_1_phi)
camera_2_cart = vf.sph2cart(1, camera_2_theta, camera_2_phi)

camera_1_vector = camera_1_cart - camera_1_r0
camera_2_vector = camera_2_cart - camera_2_r0

camera_1_t_vector = t*np.array([camera_1_vector]).T
camera_2_t_vector = t*np.array([camera_2_vector]).T

plt.figure('3D plot Camera 1 & 2 Cart')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

# ax.set_xlim([-100,100])
# ax.set_ylim([-100,100])
# ax.set_zlim([0,200])
# ax.elev = 90


ax.plot3D(xs=camera_1_t_vector[0]+camera_1_r0[0],ys=camera_1_t_vector[2]+camera_1_r0[2],zs=camera_1_t_vector[1]+camera_1_r0[1])
ax.plot3D(xs=camera_2_t_vector[0]+camera_2_r0[0],ys=camera_2_t_vector[2]+camera_2_r0[2],zs=camera_2_t_vector[1]+camera_2_r0[1])

d = vf.FindShortestDistance(camera_1_r0, camera_2_r0, camera_1_vector, camera_2_vector)
position_short_dist = vf.LocShortestDistance(camera_1_r0, camera_2_r0, camera_1_vector, camera_2_vector)

plt.show()
# %%

angles_A = np.array(f.import_bz2('A_angles_test'))
angles_B = np.array(f.import_bz2('A_angles_test'))

angles_AB = np.vstack([angles_A,angles_B])

camera1_r0 = np.array([0,0,0])
camera2_r0 = np.array([0,0,0])

cameras_r0 = [camera1_r0, camera2_r0]

for frames in range(np.shape(angles_AB)[-1]):
    print(" ", end=f"\r frame: {frames+1} ", flush=True)
    if np.any(np.isnan(angles_AB[:,frames])) == True:
        continue
    position_3D = np.zeros([3,np.shape(angles_AB)[-1]])
    position_3D[:,frames] = vf.Find3DPosition(cameras_r0, angles_AB[:,frames],args="")

# %%

import matplotlib.pyplot as plt
import numpy as np
import VectorFunctions as vf
import importlib as imp
import functions as f

imp.reload(vf)

angles_BA = np.array(f.import_bz2('test_photos_angles'))

anglesB_phi = angles_BA[0]
anglesB_theta = angles_BA[1]
anglesA_phi = angles_BA[2]
anglesA_theta = angles_BA[3]

delta = np.array([np.radians(18), np.radians(38), 0])
angles_AB = np.vstack([anglesA_phi,anglesA_theta,anglesB_phi,anglesB_theta])
#original_angles = np.full([4, np.shape(angles_AB)[-1]], np.pi/2)
#angles_AB = original_angles-angles_AB

camera1_r0 = np.array([0,0,0])
camera2_r0 = np.array([-29,1,42])

cameras_r0 = [camera1_r0, camera2_r0]

for frames in range(np.shape(angles_AB)[-1]):
    print(" ", end=f"\r frame: {frames+1} ", flush=True)
    if np.any(np.isnan(angles_AB[:,frames])) == True:
        continue        
    position_3D = np.zeros([3,np.shape(angles_AB)[-1]])
    c1 = np.zeros([3,np.shape(angles_AB)[-1]])
    c2 = np.zeros([3,np.shape(angles_AB)[-1]])

    camera1_vector, camera2_vector, c1[:,frames], c2[:,frames], position_3D[:,frames] = vf.Find3DPosition(cameras_r0, angles_AB[:,frames],delta, args="")

tlim = np.max(cameras_r0)*2
t = np.linspace(0,tlim,5000)

camera1_line = t*np.array([camera1_vector]).T[0]
camera2_line = t*np.array([camera2_vector]).T[0]

plt.figure('3D lines')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

ax.plot3D(camera1_line[0], camera1_line[2], camera1_line[1])
ax.plot3D(camera2_line[0], camera2_line[2], camera2_line[1])

print("Finished finding 3D position")

plt.figure('3D position of ball')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

ax.scatter(xs=position_3D[0,:],ys=position_3D[2,:],zs=position_3D[1,:],marker='o')
ax.scatter(xs=c1[0,:],ys=c1[2,:],zs=c1[1,:],marker='o')
ax.scatter(xs=c2[0,:],ys=c2[2,:],zs=c2[1,:],marker='o')

ax.set_xlim([-30,30])
ax.set_ylim([-0,50])
ax.set_zlim([-30,30])
ax.elev = 90
# %% TEst Final run

import matplotlib.pyplot as plt
import numpy as np
import VectorFunctions as vf
import importlib as imp
import functions as f

imp.reload(vf)

angles_BA = np.array(f.import_bz2('test_photos_angles'))

camera1_r0 = np.array([0,0,0])
camera2_r0 = np.array([-29,1,42])

I = 10

anglesB_phi = angles_BA[0] / I
anglesB_theta = angles_BA[1] / 1
anglesA_phi = angles_BA[2]/I
anglesA_theta = angles_BA[3]/1

anglesB_phi = 0
anglesB_theta = 0
anglesA_phi = 0
anglesA_theta = 0

angles_AB = np.vstack([anglesA_phi,anglesA_theta,anglesB_phi,anglesB_theta])

frames = 1

cameras_r0 = [camera1_r0, camera2_r0]
cameras_angles = angles_AB[:,frames]
delta = np.array([np.radians(18), np.radians(38), 0])

camera1_r0 = cameras_r0[0]
camera2_r0 = cameras_r0[1]

dt_A, dt_B, dp_B = delta

camera1_phi = (np.pi/2) - cameras_angles[0]
camera1_theta = (np.pi/2 + dt_A) + cameras_angles[1]

camera2_phi =  cameras_angles[2] - dp_B # -np.pi 
camera2_theta = (np.pi/2 + dt_B) - cameras_angles[3]

camera1_cart = np.array(vf.new_sph2cart(1, camera1_theta, camera1_phi, camera1_r0))
camera2_cart = np.array(vf.new_sph2cart(1, camera2_theta, camera2_phi, camera2_r0))

#camera1_vector = vf.normalised(camera1_cart - camera1_r0)
#camera2_vector = vf.normalised(camera2_cart - camera2_r0)

camera1_vector, _ = vf.EqOfLine(camera1_r0, camera1_cart)
camera2_vector, _ = vf.EqOfLine(camera2_r0, camera2_cart)

camera1_vector = camera1_vector[0]
camera2_vector = camera2_vector[0]

tlim = np.max(cameras_r0)*2
t = np.linspace(0,tlim,50000)

camera1_line = t*np.array([camera1_vector]).T + np.array([camera1_r0]).T
camera2_line = t*np.array([camera2_vector]).T + np.array([camera2_r0]).T


n = np.cross(camera1_vector,camera2_vector)

n1 = np.cross(camera1_vector, n)
n2 = np.cross(camera2_vector, n)

c1 = camera1_r0 + camera1_vector*(vf.division(np.dot((camera2_r0-camera1_r0),n2),(np.dot(camera1_vector,n2))))
c2 = camera2_r0 + camera2_vector*(vf.division(np.dot((camera1_r0-camera2_r0),n1),(np.dot(camera2_vector,n1))))

dist = np.linalg.norm(c1-c2,axis=0)

cart_position = (c1 + c2) / 2.0

plt.figure('3D position of ball')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

# for i in range(len(position_3D[-1])):
#    ax.scatter(xs=position_3D[0,i],ys=position_3D[2,i],zs=position_3D[1,i],marker='o')



# ax.scatter(camera1_line[0], camera1_line[2], camera1_line[1],marker='o')
# ax.scatter(camera2_line[0], camera2_line[2], camera2_line[1],marker='o')

ax.plot3D(camera1_line[0,[0,-1]], camera1_line[2,[0,-1]], camera1_line[1,[0,-1]])
ax.plot3D(camera2_line[0,[0,-1]], camera2_line[2,[0,-1]], camera2_line[1,[0,-1]])

ax.plot3D(camera1_r0[0], camera1_r0[2], camera1_r0[1],marker='x', color='k')
ax.plot3D(camera2_r0[0], camera2_r0[2], camera2_r0[1],marker='x', color='k')

ax.plot3D([-13, 13], [56, 56], [-27, -27], color='r')
ax.plot3D([13, 13], [56, 18], [-27, -27], color='r')
ax.plot3D([13, -13], [18, 18], [-27, -27], color='r')
ax.plot3D([-13, -13], [18, 56], [-27, -27], color='r')




ax.set_xlim([-30,30])
ax.set_ylim([0,60])
ax.set_zlim([-30,30])
ax.elev = 100
ax.azim = 300# 150


# %% Second Final run

import matplotlib.pyplot as plt
import numpy as np
import VectorFunctions as vf
import importlib as imp
import functions as f

imp.reload(vf)

angles_BA = np.array(f.import_bz2('test_photos_angles'))

camera1_r0 = np.array([0,0,0])
camera2_r0 = np.array([-29,1,42])

I = 10

anglesB_phi = angles_BA[0] / I
anglesB_theta = angles_BA[1] / 1
anglesA_phi = angles_BA[2]/I
anglesA_theta = angles_BA[3]/1

angles_AB = np.vstack([anglesA_phi,anglesA_theta,anglesB_phi,anglesB_theta])

frames = 0

cameras_r0 = [camera1_r0, camera2_r0]
cameras_angles = angles_AB[:,frames]
delta = np.array([np.radians(18), np.radians(38), 0])

camera1_r0 = cameras_r0[0]
camera2_r0 = cameras_r0[1]

dt_A, dt_B, dp_B = delta

camera1_phi = (np.pi/2) - cameras_angles[0]
camera1_theta = (np.pi/2 + dt_A) + cameras_angles[1]

camera2_phi =  -np.pi + cameras_angles[2] - dp_B
camera2_theta = (np.pi/2 + dt_B) - cameras_angles[3]

x0, y0, z0 = np.array([0,0,0]) # cameras_r0
r = 1

x1 = r * np.cos(anglesA_theta[frames]) * np.sin(anglesA_phi[frames]) + x0
z1 = r * np.sin(anglesA_theta[frames]) * np.sin(anglesA_phi[frames]) + z0
y1 = r * np.cos(anglesA_theta[frames]) + y0

x0, y0, z0 = np.array([-29,1,42])
z2 = r * np.cos(anglesB_theta[frames]) * np.sin(anglesB_phi[frames]) + x0
x2 = r * np.sin(anglesB_theta[frames]) * np.sin(anglesB_phi[frames]) + z0
y2 = r * np.cos(anglesB_theta[frames]) + y0

camera1_cart = np.hstack((x1,y1,z1))
camera2_cart = np.hstack((x2,y2,z2))
#camera1_cart = np.array(vf.new_sph2cart(1, camera1_theta, camera1_phi, camera1_r0))
#camera2_cart = np.array(vf.new_sph2cart(1, camera2_theta, camera2_phi, camera2_r0))

#camera1_vector = vf.normalised(camera1_cart - camera1_r0)
#camera2_vector = vf.normalised(camera2_cart - camera2_r0)

camera1_vector, _ = vf.EqOfLine(camera1_r0, camera1_cart)
camera2_vector, _ = vf.EqOfLine(camera2_r0, camera2_cart)

#camera1_vector = camera1_vector[0]
#camera2_vector = camera2_vector[0]

tlim = np.max(cameras_r0)*2
t = np.linspace(0,tlim,50000)

camera1_line = t*np.array([camera1_vector]).T + np.array([camera1_r0]).T
camera2_line = t*np.array([camera2_vector]).T + np.array([camera2_r0]).T


n = np.cross(camera1_vector,camera2_vector)

n1 = np.cross(camera1_vector, n)
n2 = np.cross(camera2_vector, n)

c1 = camera1_r0 + camera1_vector*(vf.division(np.dot((camera2_r0-camera1_r0),n2),(np.dot(camera1_vector,n2))))
c2 = camera2_r0 + camera2_vector*(vf.division(np.dot((camera1_r0-camera2_r0),n1),(np.dot(camera2_vector,n1))))

dist = np.linalg.norm(c1-c2,axis=0)

cart_position = (c1 + c2) / 2.0

plt.figure('3D position of ball')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

# for i in range(len(position_3D[-1])):
#    ax.scatter(xs=position_3D[0,i],ys=position_3D[2,i],zs=position_3D[1,i],marker='o')



# ax.scatter(camera1_line[0], camera1_line[2], camera1_line[1],marker='o')
# ax.scatter(camera2_line[0], camera2_line[2], camera2_line[1],marker='o')

ax.plot3D(camera1_line[0,[0,-1]], camera1_line[2,[0,-1]], camera1_line[1,[0,-1]])
ax.plot3D(camera2_line[0,[0,-1]], camera2_line[2,[0,-1]], camera2_line[1,[0,-1]])

ax.plot3D(camera1_r0[0], camera1_r0[2], camera1_r0[1],marker='x', color='k')
ax.plot3D(camera2_r0[0], camera2_r0[2], camera2_r0[1],marker='x', color='k')

ax.plot3D([-13, 13], [56, 56], [-27, -27], color='r')
ax.plot3D([13, 13], [56, 18], [-27, -27], color='r')
ax.plot3D([13, -13], [18, 18], [-27, -27], color='r')
ax.plot3D([-13, -13], [18, 56], [-27, -27], color='r')




ax.set_xlim([-30,30])
ax.set_ylim([0,60])
ax.set_zlim([-30,30])
ax.elev = 90
ax.azim = 300# 150




# %%
