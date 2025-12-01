import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

# ----- Functions -----
def bolt_area(d): return (np.pi/4)*d**2

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
    F_axial_Fz = np.full(n, L[2]/n)
    F_axial_Mx = np.where(Ixx==0, 0, L[3]*y*A/Ixx)
    F_axial_My = np.where(Iyy==0, 0, -L[4]*x*A/Iyy)
    return F_axial_Fz, F_axial_My, F_axial_Mx

def shear_loads(x, y, J, L, n,A):
    Fx = np.full(n, L[0]/n); Fy = np.full(n, L[1]/n)
    Mz = np.where(J==0, 0, L[5]*np.sqrt(x**2+y**2)*A/J)
    return Fx, Fy, Mz

def shear_stress(F_shear_x, F_shear_y, F_shear_Mz, A): return np.sqrt((F_shear_x/A)**2 + (F_shear_y/A)**2 + (F_shear_Mz/A)**2)

def normal_stress(F_axial_z, F_axial_My, F_axial_Mx, A, preload):
    total_force = F_axial_z + F_axial_My + F_axial_Mx

    # mask for positive axial forces
    mask_positive = total_force > 0

    # normal stress due to axial load (only if positive)
    sigma = np.where(mask_positive, total_force / A, 0.0)

    # add preload
    sigma += preload

    return sigma

def preload_yield(Yield,pourcentage_yield,n): return np.full(n,Yield*pourcentage_yield/100)

def preload_torque(Torque,Torque_Coefficient,Area): return (Torque/((np.sqrt((4/np.pi)*Area))*Torque_Coefficient))/Area

def VonMisesStress(tau, sigma): return np.sqrt((sigma)**2 + 3*tau**2)

def Spring_stiffness(Area,Young,Length): return Area*Young/Length

def normal_stress_stiffness(F_axial_z, F_axial_My, F_axial_Mx, F_separation, C, A, preload):
    total_force = F_axial_z + F_axial_My + F_axial_Mx

    # Before separation: slope = C, offset = preload
    normal1 = preload + (total_force * C) / A

    # After separation: The bolt takes the FULL load. 
    # The force in the bolt is just total_force.
    normal2 = total_force / A

    # mask for separation (True if NOT separated yet)
    mask_separation = total_force < F_separation
    mask_negative   = total_force < 0

    normal = np.where(mask_separation, normal1, normal2)

    # enforce no tension if total_force < 0
    normal = np.where(mask_negative, 0.0, normal)

    return normal

def length_thread_shank(L_bolt,d_m):
    if L_bolt>6*25.4:
        L_thread=2*d_m+(1/2)*25.4
    else:
        L_thread=2*d_m+(1/4)*25.4
    L_shank=L_bolt-L_thread
    return L_thread,L_shank
        
# ----- Streamlit App -----
st.title("Bolt stress analysis")
st.write("This section provides a simplified method to estimate bolt stresses. It assumes the joint has infinite stiffness and that the entire load is transmitted through the bolts, making it a conservative approximation.")

col1, col2,col3 = st.columns(3)
pattern = col1.selectbox("Bolt Pattern", ["Rectangular", "Circle", "Custom"])
d_m = col2.number_input("Nominal diameter [mm]", value=10.0, min_value=1.0, step=1.0)
pitch = col3.number_input("Pitch [mm]", value=1.0, min_value=0.0,max_value=d_m, step=1.0)
Area_min=bolt_area(d_m-0.9382*pitch) # ASME B1.13 APPENDIX B
Area_nom=bolt_area(d_m)

if pattern=="Rectangular":
    nx_col, ny_col, dx_col, dy_col = st.columns(4)
    nx = nx_col.number_input("Number of points x", value=4, min_value=1, step=1)
    ny = ny_col.number_input("Number of points y", value=3, min_value=1, step=1)
    dx = dx_col.number_input("Δx between bolts [mm]", value=100, min_value=0, step=1)
    dy = dy_col.number_input("Δy between bolts [mm]", value=100, min_value=0, step=1)
    x, y, A, n = rectangular_pattern(nx, ny, dx, dy, Area_min)
# Custom bolt pattern
elif pattern == "Custom":
    st.subheader("Custom Bolt Pattern")

    # Initialize bolt list in session state
    if "custom_bolts_list" not in st.session_state:
        st.session_state.custom_bolts_list = []

    # Input for new bolt
    col1, col2, col3 = st.columns([2,2,1])
    new_x = col1.number_input("x [mm]", value=0.0, step=1.0, key="new_x")
    new_y = col2.number_input("y [mm]", value=0.0, step=1.0, key="new_y")
    if col3.button("Add Bolt"):
        st.session_state.custom_bolts_list.append((new_x, new_y))

    # Show current bolts with delete buttons
    st.write("Current bolts (x, y) [mm]:")
    updated_bolts = []
    for i, (bx, by) in enumerate(st.session_state.custom_bolts_list):
        colx, coly, col_del = st.columns([2,2,1])
        colx.write(f"x: {bx:.1f}")
        coly.write(f"y: {by:.1f}")
        if col_del.button("Remove", key=f"remove_{i}"):
            continue  # skip adding this bolt
        updated_bolts.append((bx, by))

    # Update session state
    st.session_state.custom_bolts_list = updated_bolts

    # Stop if no bolts
    if len(st.session_state.custom_bolts_list) == 0:
        st.warning("Add at least one bolt to continue.")
        st.stop()

    # Extract arrays for computation
    x = np.array([b[0] for b in st.session_state.custom_bolts_list])
    y = np.array([b[1] for b in st.session_state.custom_bolts_list])
    n = len(x)
    A = np.full(n, Area_min)

else:
    n_col, r_col = st.columns(2)
    n_points = n_col.number_input("Points", value=12, min_value=1, step=1)
    radius = r_col.number_input("Radius [mm]", value=100.0, min_value=0.0, step=1.0)
    x, y, A, n = circle_pattern(n_points, radius, Area_min)
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
    preload=preload_torque(Torque,Torque_coefficient,Area_min)

st.subheader("Working Loads")
cols = st.columns(6)
default_loads = [0, 0, 0, 0, 0, 0]
load_labels = ["Fx [N]", "Fy [N]", "Fz [N]", "Mx [Nmm]", "My [Nmm]", "Mz [Nmm]"]

Load = [
    col.number_input(label, value=0, min_value=0, step=100)
    for label, col in zip(load_labels, cols)
]

F_shear_Fx, F_shear_Fy, F_shear_Mz = shear_loads(x, y, J, Load, n,Area_min)
F_axial_Fz, F_axial_My, F_axial_Mx = axial_loads(x, y, Ixx, Iyy, Load, n, Area_min)
tau = shear_stress(F_shear_Fx, F_shear_Fy, F_shear_Mz, Area_min)
sigma = normal_stress(F_axial_Fz, F_axial_My, F_axial_Mx, Area_min,preload)
von_mises = VonMisesStress(tau, sigma)

# ----- Improved Plot with shorter colorbar -----
fig, ax = plt.subplots(figsize=(6,6))

# Compute vmin/vmax first
vmin = np.floor(von_mises.min()/10)*10
vmax = np.ceil(von_mises.max()/10)*10

fig, ax = plt.subplots(figsize=(6,6))

# Scatter points with vmin/vmax
if Area_min < 1000:
    sc = ax.scatter(x, y, c=von_mises, cmap='viridis', s=A*2, edgecolors='k', label='Points', vmin=vmin, vmax=vmax)
else:
    sc = ax.scatter(x, y, c=von_mises, cmap='viridis', s=1000, edgecolors='k', label='Points', vmin=vmin, vmax=vmax)

# Centroid
ax.scatter(x_c, y_c, color='orange', s=100, label="Centroid", marker='+')

# Axes limits
padding = max(20, 0.2 * max(np.ptp(x), np.ptp(y)))

ax.set_xlim(np.min(x) - padding, np.max(x) + padding)
ax.set_ylim(np.min(y) - padding, np.max(y) + padding)

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
    "Normal Stress [MPa]": sigma,
    "Shear Stress [MPa]": tau,
    "Von Mises [MPa]": von_mises
})

# Display centroid separately
st.write(f"Centroid: ( {x_c:.2f} , {y_c:.2f} )")

# Display results table
st.dataframe(results_df.style.format({
    "Shear Stress [MPa]": "{:.1f}",
    "Normal Stress [MPa]": "{:.1f}",
    "Von Mises [MPa]": "{:.1f}"
}))

st.subheader("More accurate bolt calculation")
st.write("This section performs a more realistic analysis of bolt stress by accounting for both joint and bolt stiffness. The joint constant is used to determine whether the joint separates, and whether the working load reduces the preload or if the preload is fully relieved, in which case the joint carries the applied force.")

accurate_bool = st.selectbox("More accurate representation", ["Yes", "No"])

if accurate_bool=="Yes":
    col1, col2 = st.columns(2)
    Young_bolt=col1.number_input("Young Modulus bolt [MPa]", value=200.0, min_value=1.0, step=1.0)
    Length_bolt=col2.number_input("Length bolt [mm]", value=50.0, min_value=1.0, step=1.0)
    Length_thread,Length_shank=length_thread_shank(Length_bolt,d_m)
    
    col4, col5 = st.columns(2)
    Young_joint=col4.number_input("Young Modulus [MPa]", value=200.0, min_value=1.0, step=1.0)
    Area_joint=bolt_area(1.5*d_m)-bolt_area(d_m)
    Length_joint=col5.number_input("Total clamped thickness [mm]", value=10.0, min_value=1.0, step=1.0)
    
    k_shank=Spring_stiffness(Area_nom,Young_bolt,Length_shank)
    k_t=Spring_stiffness(Area_min,Young_bolt,Length_thread)
    k_bolt=(k_shank*k_t)/(k_shank+k_t)
    k_joint=Spring_stiffness(Area_joint,Young_joint,Length_joint)
    C=k_bolt/(k_bolt+k_joint)
        
    Stiffness_df = pd.DataFrame({
    "Bolt stiffness [N/mm]": [int(round(k_bolt))],
    "Joint stifness [N/mm]": [int(round(k_joint))],
    "Joint Constant C [-]": [float(C)]
    })

    # Center the values using Styler
    st.dataframe(
        Stiffness_df.style.set_properties(**{"text-align": "center"})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )
    preload_force=preload*Area_min
    normal_force=(sigma-preload)*Area_min
    force_separation=preload_force/(1-C)
    
    # Compute joint separation status
    joint_sep = np.where(normal_force > force_separation, "Yes", "No")
    normal_stress_stiff=normal_stress_stiffness(F_axial_Fz, F_axial_My, F_axial_Mx,force_separation,C,Area_min,preload)
    von_mises_stiff=VonMisesStress(tau,normal_stress_stiff)
    # Build the DataFrame
    results_separation_df = pd.DataFrame({
        "Bolt #": np.arange(1, n+1),
        "Preload Force [N]": preload_force,
        "Force Separation [N]": force_separation,
        "Normal Force [N]": normal_force,
        "Joint Separation": joint_sep,
        "Normal Stress [MPa]": normal_stress_stiff,
        "Shear Stress [MPa]": tau,
        "Von Mises [MPa]": von_mises_stiff
    })

    # Display formatted table
    st.dataframe(
        results_separation_df.style.format({
            "Preload Force [N]": "{:.1f}",
            "Force Separation [N]": "{:.1f}",
            "Normal Force [N]": "{:.1f}",
            "Normal Stress [MPa]": "{:.1f}",
            "Shear Stress [MPa]": "{:.1f}",
            "Von Mises [MPa]": "{:.1f}"
        })
    )
        # ----- Improved Plot with shorter colorbar -----
    fig, ax = plt.subplots(figsize=(6,6))

    # Compute vmin/vmax first
    vmin = np.floor(von_mises.min()/10)*10
    vmax = np.ceil(von_mises.max()/10)*10

    fig, ax = plt.subplots(figsize=(6,6))

    # Scatter points with vmin/vmax
    if Area_min < 1000:
        sc = ax.scatter(x, y, c=von_mises, cmap='viridis', s=A*2, edgecolors='k', label='Points', vmin=vmin, vmax=vmax)
    else:
        sc = ax.scatter(x, y, c=von_mises, cmap='viridis', s=1000, edgecolors='k', label='Points', vmin=vmin, vmax=vmax)

    # Centroid
    ax.scatter(x_c, y_c, color='orange', s=100, label="Centroid", marker='+')

    # Axes limits
    padding = max(20, 0.2 * max(np.ptp(x), np.ptp(y)))

    ax.set_xlim(np.min(x) - padding, np.max(x) + padding)
    ax.set_ylim(np.min(y) - padding, np.max(y) + padding)

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
    
    st.subheader("Bolt Normal Stress vs Applied Load (With Preload)")

    sep = force_separation[0]  # scalar
    preload_force_scalar = preload_force[0]  # representative bolt
    n_points = 50

    # Before separation: slope = C, offset by preload
    Fz_pre = np.linspace(0, sep, n_points)
    normal_pre = preload_force_scalar + C * Fz_pre

    # After separation: The bolt takes the ENTIRE load.
    # Therefore, Normal Force = Applied Load (Fz)
    Fz_post = np.linspace(sep, 2*sep, n_points)
    normal_post = Fz_post 

    # Combine
    Fz_total = np.concatenate([Fz_pre, Fz_post])
    normal_total = np.concatenate([normal_pre, normal_post])

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(Fz_total, normal_total, color='blue', linewidth=2, label='Normal Stress')
    ax.axvline(x=sep, color='red', linestyle='--', label='Separation Load')
    ax.set_xlabel("Applied Axial Load[N]")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Normal Force on Bolt [N]")
    ax.set_title("Bolt Normal Force vs applied axial load")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)
else:
    dummy=0