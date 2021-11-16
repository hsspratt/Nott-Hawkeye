# %% TEst Final run
%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import VectorFunctions as vf
import importlib as imp
import functions as f

imp.reload(vf)

angles_BA = np.array(f.import_bz2('test_photos_angles2'))

cameraA_r0 = np.array([0,0,0])
cameraB_r0 = np.array([-29,42,1])
cameras_r0 = [cameraA_r0, cameraB_r0]

# meausured values therefore from y axis for theta and xy for phi
anglesA_theta = angles_BA[2,:]
anglesA_phi = angles_BA[3,:]
anglesB_theta = angles_BA[0,:]
anglesB_phi = angles_BA[1,:]

angles_AB = np.vstack([anglesA_theta,anglesA_phi,anglesB_theta,anglesB_phi])

frame = 4

name = str(frame)

cameras_angles = angles_AB[:,frame]

dt_A, dt_B, dp_A, dp_B = np.array([np.radians(0), np.radians(0), np.radians(18), np.radians(38)])

cameraA_theta = np.pi/2 - cameras_angles[0]
cameraA_phi = np.pi/2 - cameras_angles[1]

cameraB_theta =  cameras_angles[2]
cameraB_phi = np.pi/2 - cameras_angles[3]
cameraB_theta = - cameraB_theta


print('camera A theta: ', cameraA_theta)
print('camera A phi: ', cameraA_phi)
print('camera B theta: ', cameraB_theta)
print('camera B phi: ', cameraB_phi)

r = 1

xA = r * np.cos(cameraA_theta + dt_A) * np.sin(cameraA_phi + dp_A) 
yA = r * np.sin(cameraA_theta + dt_A) * np.sin(cameraA_phi + dp_A) 
zA = r * np.cos(cameraA_phi + dp_A)
cameraA_vector = np.array([xA, yA, zA])
print('Cart points from A: ', xA,yA,zA)

xB = r * np.cos(cameraB_theta + dt_B) * np.sin(cameraB_phi + dp_B) 
yB = r * np.sin(cameraB_theta + dt_B) * np.sin(cameraB_phi + dp_B) 
zB = r * np.cos(cameraB_phi + dp_B) 
cameraB_vector = np.array([xB, yB, zB])
print('Cart points from B: ', xB,yB,zB)

c1, c2, dist = vf.LocShortestDistance(cameraA_r0, cameraB_r0, cameraA_vector, cameraB_vector)

position3D = c1 + c2 / 2

print(position3D)

tlim = np.max(cameras_r0)*2
t = np.linspace(0,tlim,50000)

camera1_line = t*np.array([cameraA_vector]).T + np.array([cameraA_r0]).T
camera2_line = t*np.array([cameraB_vector]).T + np.array([cameraB_r0]).T

plt.figure('3D position of ball')
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')

ax.plot3D(camera1_line[0,[0,-1]], camera1_line[1,[0,-1]], camera1_line[2,[0,-1]])
ax.plot3D(camera2_line[0,[0,-1]], camera2_line[1,[0,-1]], camera2_line[2,[0,-1]])

ax.plot3D(cameraA_r0[0], cameraA_r0[1], cameraA_r0[2],marker='x', color='k')
ax.plot3D(cameraB_r0[0], cameraB_r0[1], cameraB_r0[2],marker='x', color='k')

ax.plot3D([-13, 13], [56, 56], [-27, -27], color='r')
ax.plot3D([13, 13], [56, 18], [-27, -27], color='r')
ax.plot3D([13, -13], [18, 18], [-27, -27], color='r')
ax.plot3D([-13, -13], [18, 56], [-27, -27], color='r')

cube_size = 70
ax.set_xlim3d([-cube_size/2, cube_size/2])
ax.set_ylim3d([0, cube_size])
ax.set_zlim3d([-cube_size, 0])
ax.set_zticks([])

# name = name + name + name+ '.png'
# plt.savefig(name, dpi=600)
ax.elev = 90# 150
ax.azim = 180


# %% later ---------------------------
