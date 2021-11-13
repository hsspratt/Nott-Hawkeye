# %% Imports

from sympy import Matrix, init_printing
import sympy as sym
import sympy.printing as printing
from sympy import Integral, Matrix, pi, pprint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpmath as mp
import functions as f

# %% All Functions and Definitions relating to working with vectors

def division(n1, n2):
    try:
        return n1/n2
    except ZeroDivisionError:
        return 0

def normalised(v):
    """Normalises vector - ensure vector direction is of unit length

    Parameters
    ----------
    v : np.array (N,1)
        np.array of vector direction

    Returns
    -------
    norm_v : np.array (N,1)
        Normalised vector - of unit length 1
    """    

    norm_v = v / (np.sqrt(np.sum(v**2)))

    return norm_v


def EqOfLine(r0,r1):
    """Creates a vector equation of the line in the form r = vt + r_0

    Parameters
    ----------
    r0 : np.array (N,1)
        Numpy array of a origin point (can also be a point which lies on the line)
    r1 : np.array (N,1)
        Numpy array of a point which lies on the line
        
    - N is the dimensions of the vector

    Returns
    -------
    vector : np.array (N,1)
        vector that indicates the direction of the line - which is normalised
    r0 : np.array (N,1)
        Rereturns r0, so that in can be easily found
    """    
    vector = normalised(np.array([r1 - r0]))
    r0 = np.array([r0])

    return [vector, r0]

def FindShortestDistance(r1, r2, v1, v2):
    """Finds the shortest distance between two vectors

    Parameters
    ----------
    r1 : np.array (N,1)
        Numpy array of a point which lies on the line 1
    r2 : np.array (N,1)
        Numpy array of a point which lies on the line 2
    v1 : np.array (N,1)
        vector that indicates the direction of the line 1
    v2 : np.array (N,1)
        vector that indicates the direction of the line 2

    Returns
    -------
    distance : float
        simply the shortest distance between the two lines
    """    
    n = np.cross(v1,v2)

    if n.all == 0:
        # folling eq only works IF parallel
        distance = np.sqrt((np.cross((r2-r1),v1))**2/(np.linalg.norm(v2,axis=0))**2)
        print("The two lines are parallel")
    else:
        # folling eq only works if NOT parallel
        distance = (np.dot(n,(r1-r2)))/(np.linalg.norm(n,axis=0))
        print("The two lines are not parallel")

    return distance

def LocShortestDistance(r1, r2, v1, v2):
    """Finds the location of the shortest distance between two lines on the respective lines

    Parameters
    ----------
    r1 : np.array (N,1)
        Numpy array of a point which lies on the line 1
    r2 : np.array (N,1)
        Numpy array of a point which lies on the line 2
    v1 : np.array (N,1)
        vector that indicates the direction of the line 1
    v2 : np.array (N,1)
        vector that indicates the direction of the line 2

    Returns
    -------
    c1 : np.array (N,1)
        Location on line 1 where it is at a shortest distnce to line 2
    c2 : np.array (N,1)
        Location on line 2 where it is at a shortest distnce to line 1
    dist : float
        simply the shortest distance between the two lines 

    - should be noted dist is less accurate then dedicated function FindShortestDistance

    """    

    n = np.cross(v1,v2)

    n1 = np.cross(v1, n)
    n2 = np.cross(v2, n)

    c1 = r1 + v1*(division(np.dot((r2-r1),n2),(np.dot(v1,n2))))
    c2 = r2 + v2*(division(np.dot((r1-r2),n1),(np.dot(v2,n1))))

    dist = np.linalg.norm(c1-c2,axis=0)

    return [c1, c2, dist]

def Polar2Vector(r0, angle, axis, camera):
    """Converts a polar line - 1D line in 2D space (a plane) into a 1D vector line in 3D space

    Parameters
    ----------
    r0 : np.array (N,1)
        Numpy array of a origin point (can also be a point which lies on the line)
    angle : radians (-π ≤ θ ≤ π)
        Angle from a line normal to the lens to the angle made with the pixel
    axis : string ("xz" or "yz")
        Specify the axis in question - "xs" is the horizontal axis and "yz" is the vertical
    camera : string ("1" or "2")
        Specify the camera in question, "1" is at the orgin and "2" is at (x1, 0, z1)

    Returns
    -------
    r0 : np.array (N,1)
        Numpy array of a origin point (can also be a point which lies on the line)
    vector : np.array (N,1)
        vector that indicates the direction of the line
    """   
    if np.any(angle) == 0:
        angle = 0.001

    if camera == "1":
        if axis == "xz":
            vector = np.array([[1, 0, mp.cot(angle)]])
            return r0, vector
        if axis == "yz":
            vector = np.array([[0, 1, mp.cot(angle)]])
            return r0, vector
        else:
            print("Axis not specified correctly")
    if camera=="2":
        if axis == "xz":
            vector = np.array([[1, 0, mp.tan(angle)]])
            return r0, vector
        if axis == "yz":
            vector = np.array([[0, 1, mp.tan(angle)]])
            return r0, vector
        else:
            print("Axis not specified correctly")


def sph2cart(r, theta, phi):
    """Converts spherical co-ordinates (r,θ,φ) into cartesian (x,y,z)

    Parameters
    ----------
    r : float
        radial distance
    theta : radians
        Angle from a line normal to the lens to the angle made with the pixel (xz plane)
    phi : radians
        Angle from a line normal to the lens to the angle made with the pixel (yz plane)

    Returns
    -------
    [x,y,z] : np.array of floats
        Denotes the x, y and z co-ordinate
    """    

    x = r * mp.sin(theta) * mp.cos(phi)
    y = r * mp.sin(theta) * mp.sin(phi)
    z = r * mp.cos(theta)

    return [x,y,z]

def Find3DPosition(cameras_r0, cameras_angles, args):
    """Takes the inputs from the cameras in terms of positions and angles
       generate a 3D position of the ball

    Parameters
    ----------
    cameras_r0 : turple of np.arrays
        Two numpy arrays for the two initial positions of the cameras
    cameras_angles : turple of np.arrays
        Four angles theta and phi from camera 1 and the same from camera 2
    args : string
        if "lines" chosen it returns the whole line instead of the individual point

    Returns
    -------
    cart_position
        Returns the 3D position of the ball in cartesian co-ordinates
    """    
    camera1_r0 = cameras_r0[0]
    camera2_r0 = cameras_r0[1]

    camera1_theta = cameras_angles[0]
    camera1_phi = cameras_angles[1]
    camera2_theta = cameras_angles[2]
    camera2_phi = cameras_angles[3]

    _,camera1_vector_xz = Polar2Vector(camera1_r0, camera1_theta, axis="xz", camera="1")
    _,camera1_vector_yz = Polar2Vector(camera1_r0, camera1_phi, axis="yz", camera="1")
    _,camera2_vector_xz = Polar2Vector(camera2_r0, camera2_theta, axis="xz", camera="2")
    _,camera2_vector_yz = Polar2Vector(camera2_r0, camera2_phi, axis="yz", camera="2")

    camera1_cart = sph2cart(1, camera1_theta, camera1_phi)
    camera2_cart = sph2cart(1, camera2_theta, camera2_phi)

    camera1_vector = normalised(camera1_cart - camera1_r0)
    camera2_vector = normalised(camera2_cart - camera2_r0)

    tlim = np.max(camera1_r0*2)
    t = np.linspace(0,tlim,5000)
    
    camera1_line = t*np.array([camera1_vector]).T
    camera2_line = t*np.array([camera2_vector]).T

    if args == "Line":
        return [camera1_line, camera2_line], [camera1_vector, camera2_vector]

    rel_position_shortest = LocShortestDistance(camera1_r0, camera2_r0, camera1_vector, camera2_vector)
    rel_position_shortest = [rel_position_shortest[0].astype(np.float64), rel_position_shortest[1].astype(np.float64)]
    #print(rel_position_shortest)
    # Z = np.matrix(rel_position_shortest.tolist(), dtype=float)
    cart_position = np.average(rel_position_shortest[0:2])

    return cart_position

# def Camera2_Camera1_axis(phi,z1,x1):
#     """Converts the co-ordinates from that of camera 2 to camera 1 - therefore one co-ordinate system

#     Parameters
#     ----------
#     phi : radian
#         Angle from a line normal to the lens to the angle made with the pixel
#     z1 : float
#         z position of camera 2
#     x1 : [type]
#         x position of camera 2

#     Returns
#     -------
#     r0 : np.array
#         [description]
#     """    
#     r0 = np.array([z1 - np.tan(phi)*x1, 0])
#     vector = np.array([[np.tan(phi), 1]])
#     return r0, vector

# def Camera1toCamera2_1(phi,z1,x1):
#     r0 = np.array([x1, z1])
#     vector = np.array([[np.tan(phi), 1]])
#     return r0, vector



# %%
