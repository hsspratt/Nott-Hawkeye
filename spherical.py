# %%
# %matplotlib widget
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a sphere
r = np.linspace(0,1,100)
pi = np.pi
cos = np.cos
sin = np.sin
# phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
# phi, theta = np.mgrid[0:10, 0:10]
phi,theta = np.array([0,0])
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

#Import data
# data = np.genfromtxt('leb.txt')
# xx, yy, zz = np.hsplit(data, 3) 

#Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(
#     x, y, z,  rstride=1, cstride=1, color='c', alpha=1, linewidth=0)

# ax.scatter(xx,yy,zz,color="r",s=100)

ax.scatter(x, y, z)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect("auto")
plt.tight_layout()
plt.show()
# %%
