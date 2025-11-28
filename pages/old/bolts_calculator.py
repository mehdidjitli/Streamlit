#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 20:15:41 2025

@author: mehdidjitli
"""

import numpy as np
import matplotlib.pyplot as plt

def circle_pattern(n,r,Area):
    x=np.zeros(n)
    y=np.zeros(n)
    dtheta=2*np.pi/n
    A=np.zeros(n)+Area
    for i in range(0,len(x)):
        x[i]=r*np.cos(i*dtheta)
        y[i]=r*np.sin(i*dtheta)
    return x,y,A,n

def rectangular_pattern(nx, ny, dx, dy, Area):
    # Coordinates centered around 0
    x_coords = np.linspace(-(nx-1)*dx/2, (nx-1)*dx/2, nx)
    y_coords = np.linspace(-(ny-1)*dy/2, (ny-1)*dy/2, ny)

    # Create full grid
    X, Y = np.meshgrid(x_coords, y_coords)

    # Flatten to get lists of coordinates
    x = X.flatten()
    y = Y.flatten()

    A = np.full_like(x, Area, dtype=float)
    n=nx*ny
    return x, y, A, n

def centroid(x,y):
    x_c=np.average(x)
    y_c=np.average(y)
    return x_c, y_c

def Inertia(x,y,A):
    Ixx=y**2*A
    Iyy=x**2*A
    J=Ixx+Iyy
    return Ixx,Iyy,J

def axial_loads(n,Ixx,Iyy):
    
    F_axial_Mx=np.zeros(n)
    F_axial_My=np.zeros(n)
    F_axial_Fz=np.zeros(n)
    
    for i in range(0,n):
        F_axial_Fz[i]=Load[2]/n
        if Ixx[i]==0:
            F_axial_Mx[i]=0
        elif Iyy[i]==0:
            F_axial_My[i]=0
        else:
            F_axial_Mx[i]=Load[3]*y[i]/Ixx[i]
            F_axial_My[i]=Load[4]*x[i]/Iyy[i]
            
    return F_axial_Fz, F_axial_My, F_axial_Mx

def shear_loads(n,x,y,J):
    F_shear_Fx=np.zeros(n)
    F_shear_Fy=np.zeros(n)
    F_shear_Mz=np.zeros(n)
    for i in range(0,n):
        F_shear_Fx[i]=Load[0]/n
        F_shear_Fy[i]=Load[1]/n
        if J[i]==0:
            F_shear_Mz[i]=0
        else:
            F_shear_Mz[i]=Load[5]*np.sqrt(x[i]**2+y[i]**2)/J[i]
    return F_shear_Fx,F_shear_Fy,F_shear_Mz

def shear_stress(F_shear_Fx,F_shear_Fy,F_shear_Mz,Area):
    tau_xx=F_shear_Fx/Area
    tau_yy=F_shear_Fy/Area
    tau_xy=F_shear_Mz/Area
    tau=np.sqrt(tau_xx**2+tau_yy**2+tau_xy**2)
    return tau

def normal_stress(F_axial_Fz,F_axial_My, F_axial_Mx):
    sigma_normal=F_axial_Fz/Area
    sigma_bending_y=F_axial_My/Area
    sigma_bending_x=F_axial_Mx/Area
    sigma=sigma_normal+sigma_bending_y+sigma_bending_x
    return sigma

def VonMises(tau,sigma):
    return np.sqrt(sigma**2+3*tau**2)

#########TEST#########

# GEOMETRY DEFINITION

Area=70
x,y,A,n= rectangular_pattern(6,3,1,1,Area)
print(x,y)
print(len(x))
x_c,y_c=centroid(x,y)

# PLOT

plt.scatter(x, y, label="Points")
plt.scatter(x_c, y_c, color='red', label="Centroid", zorder=5,marker='+')
plt.axis('equal')  # Ensures same scale on both axes
plt.title("Circular Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# LOADS EXTRACTION

Ixx,Iyy,J=Inertia(x,y,A)
Load=[50000,1200,1200,1000,1000,1000]
F_shear_Fx,F_shear_Fy,F_shear_Mz=shear_loads(n,x,y,J)
F_axial_Fz,F_axial_My, F_axial_Mx=axial_loads(n,Ixx,Iyy)

# PRINT LOADS

print(x)
print(y)
print("Shear x [N]:",F_shear_Fx)
print("Shear y [N]:",F_shear_Fy)
print("Normal z [N]:",F_axial_Fz)
print("Normal x [N]:",F_axial_Mx)
print("Normal y [N]:",F_axial_My)
print("Shear Mz [N]:",F_shear_Mz)

# STRESS

tau=shear_stress(F_shear_Fx,F_shear_Fy,F_shear_Mz,Area)
sigma=normal_stress(F_axial_Fz,F_axial_My, F_axial_Mx)
VonMises=VonMises(tau,sigma)

# PRINT STRESS

print("Shear Stress [MPa]:", tau)
print("Normal Stress [MPa]:",sigma)
print("Von Mises Stress [MPa]:",VonMises)



