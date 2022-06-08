import numpy as np
from scipy.spatial.transform import Rotation

def norm_3d(a):
    return np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])

def sample_on_unit_sphere(rng):
    '''
    Algo from http://corysimon.github.io/articles/uniformdistn-on-sphere/
    '''
    v = np.zeros(3)
    while norm_3d(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()
        v[2] = rng.normal()
    
    v = v / norm_3d(v)
    return v

def norm_2d(a):
    return np.sqrt(a[0]*a[0]+a[1]*a[1])

def sample_on_unit_circle(rng):
    v = np.zeros(2)
    while norm_2d(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()
    
    v = v / norm_2d(v)
    return v

def rotation_between_vec(a, b): # from a to b
    a = a / norm_3d(a)
    b = b / norm_3d(b)
    axis = np.cross(a, b)
    axis = axis / norm_3d(axis) # norm might be 0
    angle = np.arccos(a @ b)
    R = Rotation.from_rotvec( axis * angle )
    return R


def wxyz_to_xyzw(q):
    return np.concatenate([ q[1:4], q[0:1] ])

def xyzw_to_wxyz(q):
    return np.concatenate([ q[3:4], q[0:3] ])

def rotate_2d_vec_by_angle(vec, theta):
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return rot_mat @ vec