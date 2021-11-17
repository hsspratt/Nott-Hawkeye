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
cameras_r0 = [camera1_r0, camera2_r0]

I = 10

anglesB_phi = [0,0]
anglesB_theta = [0,0]
anglesA_phi = [0,0]
anglesA_theta = [0,0]

angles_AB = np.vstack([anglesA_phi,anglesA_theta,anglesB_phi,anglesB_theta])

frames = 0
"""
cameras_r0 = [camera1_r0, camera2_r0]
camera1_r0 = cameras_r0[0]
camera2_r0 = cameras_r0[1]
"""

cameras_angles = angles_AB[:,frames]
delta = np.array([np.radians(0), np.radians(0), 0])

dt_A, dt_B, dp_B = delta

camera1_phi = (np.pi/2) - cameras_angles[0]
camera1_theta = (np.pi/2 + dt_A) - cameras_angles[1]

camera2_phi =  cameras_angles[2] - dp_B
camera2_theta = (np.pi/2 + dt_B) - cameras_angles[3]

camera1_cart = np.array(vf.new_sph2cart(1, camera1_theta, camera1_phi, camera1_r0))
camera2_cart = np.array(vf.new_sph2cart(1, camera2_theta, camera2_phi, camera2_r0))

#camera1_vector = vf.normalised(camera1_cart - camera1_r0)
#camera2_vector = vf.normalised(camera2_cart - camera2_r0)

camera1_vector, _ = vf.EqOfLine(camera1_r0, camera1_cart)
camera2_vector, _ = vf.EqOfLine(camera2_r0, camera2_cart)

# camera1_vector = camera1_vector
# camera2_vector = camera2_vector

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


# %%
