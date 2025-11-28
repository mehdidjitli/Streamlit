import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

# ----- Functions -----
def circle_pattern(n, r, Area):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, np.full(n, Area), n

def rectangular_pattern(nx, ny, dx, dy, Area):
    x, y = np.meshgrid(np.linspace(-(nx-1)*dx/2, (nx-1)*dx/2, nx),
                       np.linspace(-(ny-1)*dy/2, (ny-1)*dy/2, ny))
    return x.flatten(), y.flatten(), np.full(nx*ny, Area), nx*ny

def centroid(x, y): return np.mean(x), np.mean(y)
def Inertia(x, y, A): Ixx=np.sum(y**2*A); Iyy=np.sum(x**2*A); J=Ixx+Iyy; return Ixx, Iyy, J
def axial_loads(x, y, Ixx, Iyy, L, n, A):
    Fz = np.full(n, L[2]/n)
    Mx = np.where(Ixx==0, 0, L[3]*y/Ixx)
    My = np.where(Iyy==0, 0, -L[4]*x/Iyy)
    return Fz, My, Mx

def shear_loads(x, y, J, L, n):
    Fx = np.full(n, L[0]/n); Fy = np.full(n, L[1]/n)
    Mz = np.where(J==0, 0, L[5]*np.sqrt(x**2+y**2)/J)
    return Fx, Fy, Mz

def shear_stress(Fx, Fy, Mz, A): return np.sqrt((Fx/A)**2 + (Fy/A)**2 + (Mz/A)**2)
def normal_stress(Fz, My, Mx, A): return Fz/A + My/A + Mx/A
def preload_yield(Yield,pourcentage_yield,n): return np.full(n,Yield*pourcentage_yield/100)
def preload_torque(Torque,Torque_Coefficient,Area): return (Torque/((np.sqrt((4/np.pi)*Area))*Torque_Coefficient))/Area
def VonMisesStress(tau, sigma,preload): return np.sqrt((sigma+preload)**2 + 3*tau**2)

# ----- Streamlit App -----
st.title("Bolt stress analysis")
st.write("This section provides a simplified method to estimate bolt stresses. It assumes the joint has infinite stiffness and that the entire load is transmitted through the bolts, making it a conservative approximation.")

col1, col2 = st.columns(2)
pattern = col1.selectbox("Bolt Pattern", ["Rectangular", "Circle"])
Area = col2.number_input("Section Area [mm2]", value=100.0, min_value=1.0, step=1.0)

if pattern=="Rectangular":
    nx_col, ny_col, dx_col, dy_col = st.columns(4)
    nx = nx_col.number_input("Number of points x", value=4, min_value=1, step=1)
    ny = ny_col.number_input("Number of points y", value=3, min_value=1, step=1)
    dx = dx_col.number_input("dx between bolts [mm]", value=100, min_value=0, step=1)
    dy = dy_col.number_input("dy between bolts [mm]", value=100, min_value=0, step=1)
    x, y, A, n = rectangular_pattern(nx, ny, dx, dy, Area)
else:
    n_col, r_col = st.columns(2)
    n_points = n_col.number_input("Points", value=12, min_value=1, step=1)
    radius = r_col.number_input("Radius [mm]", value=100.0, min_value=0.0, step=1.0)
    x, y, A, n = circle_pattern(n_points, radius, Area)
x_c, y_c = centroid(x, y)
Ixx, Iyy, J = Inertia(x, y, A)

# Create DataFrame
inertia_df = pd.DataFrame({
    "Ixx [mm4]": [int(round(Ixx))],
    "Iyy [mm4]": [int(round(Iyy))],
    "J [mm4]": [int(round(J))]
})

# Center the values using Styler
st.subheader("Inertia Properties")
st.dataframe(
    inertia_df.style.set_properties(**{"text-align": "center"})
             .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
)
st.subheader("Preload")
col3, col4, col5 = st.columns(3)
preload_selection = col3.selectbox("Preload Calculation", ["% Yield", "By Torque"])
if preload_selection=="% Yield":
    Yield = col4.number_input("Yield [MPa]", value=240.0, min_value=0.0, step=1.0)
    pourcentage_yield = col5.number_input("% Yield", value=60.0, min_value=0.0, step=1.0)
    preload=preload_yield(Yield,pourcentage_yield,n)
else:
    Torque = col4.number_input("Torque [Nmm]", value=300.0, min_value=0.0, step=1.0)
    Torque_coefficient = col5.number_input("Torque coefficient", value=0.2, min_value=0.0, step=0.1)
    preload=preload_torque(Torque,Torque_coefficient,Area)

st.subheader("Working Loads")
cols = st.columns(6)
default_loads = [0, 0, 0, 0, 0, 0]

load_labels = ["Fx [N]", "Fy [N]", "Fz [N]", "Mx [Nmm]", "My [Nmm]", "Mz [Nmm]"]

Load = [
    col.number_input(label, value=0, min_value=0, step=100)
    for label, col in zip(load_labels, cols)
]

F_shear_Fx, F_shear_Fy, F_shear_Mz = shear_loads(x, y, J, Load, n)
F_axial_Fz, F_axial_My, F_axial_Mx = axial_loads(x, y, Ixx, Iyy, Load, n, Area)
tau = shear_stress(F_shear_Fx, F_shear_Fy, F_shear_Mz, Area)
sigma = normal_stress(F_axial_Fz, F_axial_My, F_axial_Mx, Area)
von_mises = VonMisesStress(tau, sigma, preload)

# ----- Improved Plot with shorter colorbar -----
fig, ax = plt.subplots(figsize=(6,6))

# Compute vmin/vmax first
vmin = np.floor(von_mises.min()/10)*10
vmax = np.ceil(von_mises.max()/10)*10

fig, ax = plt.subplots(figsize=(6,6))

# Scatter points with vmin/vmax
if Area < 1000:
    sc = ax.scatter(x, y, c=von_mises, cmap='viridis', s=A*2, edgecolors='k', label='Points', vmin=vmin, vmax=vmax)
else:
    sc = ax.scatter(x, y, c=von_mises, cmap='viridis', s=1000, edgecolors='k', label='Points', vmin=vmin, vmax=vmax)

# Centroid
ax.scatter(x_c, y_c, color='orange', s=100, label="Centroid", marker='+')

# Axes limits
try:
    ax.set_xlim(np.min(x)-dx, np.max(x)+dx)
    ax.set_ylim(np.min(y)-dy, np.max(y)+dy)
except:
    ax.set_xlim(np.min(x)-radius*0.2, np.max(x)+radius*0.2)
    ax.set_ylim(np.min(y)-radius*0.2, np.max(y)+radius*0.2)

# Number each bolt
for i, (xi, yi) in enumerate(zip(x, y), start=1):
    ax.text(xi, yi, str(i), color='red', fontsize=10, ha='center', va='center')

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_title("Bolt Pattern Von Mises Stress")
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

# Colorbar
cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label('Von Mises Stress [MPa]')
cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
cbar.ax.yaxis.get_offset_text().set_visible(False)

st.pyplot(fig)


st.subheader("Results")

# Create results table
results_df = pd.DataFrame({
    "Bolt #": np.arange(1, n+1),
    "Shear Stress [MPa]": tau,
    "Normal Stress [MPa]": sigma,
    "Preload Stress [MPa]": preload,
    "Von Mises [MPa]": von_mises
})

# Display centroid separately
st.write(f"Centroid: ( {x_c:.2f} , {y_c:.2f} )")

# Display results table
st.dataframe(results_df.style.format({
    "Shear Stress [MPa]": "{:.1f}",
    "Normal Stress [MPa]": "{:.1f}",
    "Preload Stress [MPa]": "{:.1f}",
    "Von Mises [MPa]": "{:.1f}"
}))

