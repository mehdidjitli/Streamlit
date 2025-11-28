import streamlit as st

st.title("🔧 My Engineering Toolbox")
st.write("Welcome to my first Streamlit app!")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----- Fonctions -----
def circle_pattern(n, r, Area):
    x = np.zeros(n)
    y = np.zeros(n)
    dtheta = 2*np.pi/n
    A = np.zeros(n) + Area
    for i in range(n):
        x[i] = r * np.cos(i*dtheta)
        y[i] = r * np.sin(i*dtheta)
    return x, y, A, n

def rectangular_pattern(nx, ny, dx, dy, Area):
    x_coords = np.linspace(-(nx-1)*dx/2, (nx-1)*dx/2, nx)
    y_coords = np.linspace(-(ny-1)*dy/2, (ny-1)*dy/2, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    x = X.flatten()
    y = Y.flatten()
    A = np.full_like(x, Area, dtype=float)
    n = nx * ny
    return x, y, A, n

def centroid(x, y):
    return np.average(x), np.average(y)

def Inertia(x, y, A):
    Ixx = y**2 * A
    Iyy = x**2 * A
    J = Ixx + Iyy
    return Ixx, Iyy, J

def axial_loads(x, y, Ixx, Iyy, Load, n, Area):
    F_axial_Mx = np.zeros(n)
    F_axial_My = np.zeros(n)
    F_axial_Fz = np.zeros(n)
    
    for i in range(n):
        F_axial_Fz[i] = Load[2]/n
        F_axial_Mx[i] = 0 if Ixx[i]==0 else Load[3]*y[i]/Ixx[i]
        F_axial_My[i] = 0 if Iyy[i]==0 else Load[4]*x[i]/Iyy[i]
    
    return F_axial_Fz, F_axial_My, F_axial_Mx

def shear_loads(x, y, J, Load, n):
    F_shear_Fx = np.zeros(n)
    F_shear_Fy = np.zeros(n)
    F_shear_Mz = np.zeros(n)
    
    for i in range(n):
        F_shear_Fx[i] = Load[0]/n
        F_shear_Fy[i] = Load[1]/n
        F_shear_Mz[i] = 0 if J[i]==0 else Load[5]*np.sqrt(x[i]**2+y[i]**2)/J[i]
    
    return F_shear_Fx, F_shear_Fy, F_shear_Mz

def shear_stress(F_shear_Fx, F_shear_Fy, F_shear_Mz, Area):
    tau_xx = F_shear_Fx / Area
    tau_yy = F_shear_Fy / Area
    tau_xy = F_shear_Mz / Area
    tau = np.sqrt(tau_xx**2 + tau_yy**2 + tau_xy**2)
    return tau

def normal_stress(F_axial_Fz, F_axial_My, F_axial_Mx, Area):
    sigma_normal = F_axial_Fz / Area
    sigma_bending_y = F_axial_My / Area
    sigma_bending_x = F_axial_Mx / Area
    sigma = sigma_normal + sigma_bending_y + sigma_bending_x
    return sigma

def VonMisesStress(tau, sigma):
    return np.sqrt(sigma**2 + 3*tau**2)

# ----- Interface Streamlit -----
st.title("Analyse de contraintes sur pattern de points")

pattern_type = st.selectbox("Type de pattern", ["Rectangular", "Circle"])

Area = st.number_input("Section Area", value=70.0)

if pattern_type == "Rectangular":
    nx = st.number_input("Nombre de points en x", value=6, min_value=1)
    ny = st.number_input("Nombre de points en y", value=3, min_value=1)
    dx = st.number_input("Distance entre points en x", value=1.0)
    dy = st.number_input("Distance entre points en y", value=1.0)
    x, y, A, n = rectangular_pattern(nx, ny, dx, dy, Area)
else:
    n_points = st.number_input("Nombre de points", value=12, min_value=1)
    radius = st.number_input("Rayon", value=5.0)
    x, y, A, n = circle_pattern(n_points, radius, Area)

st.subheader("Charges appliquées")
Load_Fx = st.number_input("Force de cisaillement Fx", value=50000.0)
Load_Fy = st.number_input("Force de cisaillement Fy", value=1200.0)
Load_Fz = st.number_input("Force axiale Fz", value=1200.0)
Load_Mx = st.number_input("Moment Mx", value=1000.0)
Load_My = st.number_input("Moment My", value=1000.0)
Load_Mz = st.number_input("Moment Mz", value=1000.0)

Load = [Load_Fx, Load_Fy, Load_Fz, Load_Mx, Load_My, Load_Mz]

# Calculs
x_c, y_c = centroid(x, y)
Ixx, Iyy, J = Inertia(x, y, A)
F_shear_Fx, F_shear_Fy, F_shear_Mz = shear_loads(x, y, J, Load, n)
F_axial_Fz, F_axial_My, F_axial_Mx = axial_loads(x, y, Ixx, Iyy, Load, n, Area)
tau = shear_stress(F_shear_Fx, F_shear_Fy, F_shear_Mz, Area)
sigma = normal_stress(F_axial_Fz, F_axial_My, F_axial_Mx, Area)
von_mises = VonMisesStress(tau, sigma)

# Affichage graphique
fig, ax = plt.subplots()
ax.scatter(x, y, label="Points")
ax.scatter(x_c, y_c, color='red', label="Centroid", zorder=5, marker='+')
ax.set_aspect('equal')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Pattern de points")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Affichage résultats
st.subheader("Résultats")
st.write("Centroid: ", (x_c, y_c))
st.write("Shear Stress [MPa]:", tau)
st.write("Normal Stress [MPa]:", sigma)
st.write("Von Mises Stress [MPa]:", von_mises)
st.write("Shear Forces Fx:", F_shear_Fx)
st.write("Shear Forces Fy:", F_shear_Fy)
st.write("Axial Forces Fz:", F_axial_Fz)
st.write("Axial Forces Mx:", F_axial_Mx)
st.write("Axial Forces My:", F_axial_My)
st.write("Moments Mz:", F_shear_Mz)

    