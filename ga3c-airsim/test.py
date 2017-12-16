import numpy as np
from math import radians, cos, sin
from projection import *
g = 9.8
def move(roll, pitch, yaw, thrust, mass):
    A = radians(yaw)
    B = radians(pitch)
    C = radians(roll)
    f = np.matrix([0,0,float(thrust)/float(mass), 1])
    R = np.matrix([[cos(A)*cos(B), cos(A)*sin(B)*sin(C)-sin(A)*cos(C), cos(A)*sin(B)*cos(C) + sin(A)*sin(C), 0],
                   [sin(A)*cos(B), sin(A)*sin(B)*sin(C)+cos(A)*cos(C), sin(A)*sin(B)*cos(C) - cos(A)*sin(C), 0],
                   [-sin(B),       cos(B)*sin(C),                      cos(B)*cos(C),                        0],
                   [0,             0,                                  0,                                    1]])
    a = R*np.transpose(f)
    a = np.transpose(a[:3])
    return a
