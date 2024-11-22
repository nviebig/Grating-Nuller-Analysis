import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn  # for J_n(x)

def U(z,U_0,wavelength,periode_gratting,q_a,theta,q_b,a):

    """
    Calculate the field U at a distance z.
    ---------------------------------------
    Parameters:
    z (float): Distance at which the field is calculated.
    U_0 (complex): Initial field amplitude.
    wavelength (float): Wavelength of the light.
    periode_gratting (float): Period of the grating.
    q_a (int): Diffraction order for the lower beam.
    theta (float): Angle of incidence.
    q_b (int): Diffraction order for the upper beam.
    a (float): Parameter related to the grating.
    
    Returns:
    complex: The calculated field U at distance z.
    """

    k = 2*np.pi/wavelength
    h_0 = -1j*np.exp(1j*k*z)/(wavelength*z)
    alpha = k/(2*z)
    n = 1.5  
    s = 0.5e-6
    m = 2 * np.pi *(n-1)*s/wavelength
    
    a_q = 2 * np.pi * q_a / periode_gratting + k * np.sin(theta)
    b_q = 2 * np.pi * q_b / periode_gratting - k * np.sin(theta)

    aq=(a_q**2)/(8*alpha)
    bq=(b_q**2)/(8*alpha)

    k_a = ((a_q/4)*jn(q_a,m/2)*np.exp(-1j*((aq)-q_a*np.pi/4))*(jn(1/2,aq)+1j*jn(-1/2,aq)))*np.exp( 1j*k*np.sin(theta)*a)
    k_b = ((b_q/4)*jn(q_b,m/2)*np.exp(-1j*((bq)-q_b*np.pi/4))*(jn(1/2,bq)+1j*jn(-1/2,bq)))*np.exp(-1j*k*np.sin(theta)*a)

    return U_0*h_0*np.sqrt((np.pi/alpha)**3)*(k_a+k_b)


z = 0.1 # Distance at which the field is calculated (0.1 m)
U_0 = 1  # Initial field amplitude (1)
wavelength = 633e-9 # Wavelength of the light (633 nm = 633e-9 m)
periode_gratting = 3.3e-6  # Grating period in meters (3.3 micron=3.3e-6 m)
q_a= -1 # Diffraction order for the lower beam (-1)
theta = np.arcsin(wavelength/periode_gratting)
 # Angle of incidence (0.19106029214617076 rad)
q_b= 1 # Diffraction order for the upper beam (1)


a_fkt_min = lambda n: periode_gratting/2 * (3/4 + n)
a_pick_min = a_fkt_min(0)
a_pick_max = a_pick_min - ( a_fkt_min(1)- a_fkt_min(0))/2

z_pick = 0.1 # Distance at which the field is calculated (0.1 m)

f = lambda z,a: np.abs(U(z,U_0,wavelength,periode_gratting,q_a,theta,q_b,a))**2
a = np.linspace(-2, 2, 1000)*1e-6


fig, axs = plt.subplots(2, 2, figsize=(20, 12))
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

# First plot
ax1.set_title("Intensity of the field U at a distance z = {}".format(z_pick))
ax1.set_xlabel("a (microns)")
ax1.set_ylabel("Intensity")
ax1.plot(a, f(z_pick,a))
ax1.axhline(y=0, color='r', linestyle='--')
ax1.axvline(x=a_pick_min, color='r', linestyle='--', label="a={}, f(a) = {}".format(round(a_pick_min,9),round(f(z,a_pick_min),9)))
ax1.set_yticks(np.arange(0, 1.1, 0.1))  # Generate ticks between -0.1 and 1 with step 0.1
ax1.grid()
ax1.legend(loc="upper right")

# Second plot
z_vals = np.linspace(0.01,z_pick,1000)
ax2.plot(z_vals, f(z_vals, a_pick_min), color='r', label="a = {}".format(round(a_pick_min)),linewidth=5)
ax2.set_title("Intensity of the field U at value a = {}".format(round(a_pick_min,9)))
ax2.set_xlabel("z (m)")
ax2.set_ylabel("Intensity")
ax2.set_yticks(np.arange(-0.1, 1.1, 0.1))  # Generate ticks between -0.1 and 1 with step 0.1
ax2.grid()

# Third plot
ax3.set_title("Intensity of the field U at a distance z = {}".format(z_pick))
ax3.set_xlabel("a (microns)")
ax3.set_ylabel("Intensity")
ax3.plot(a, f(z_pick,a))
ax3.axhline(y=max(f(z_pick,a)), color='g', linestyle='--')
ax3.axvline(x=a_pick_max, color='g', linestyle='--', label="a={}, f(a) = {}".format(round(a_pick_max,9),round(f(z,a_pick_max),9)))
ax3.legend(loc="upper right")
ax3.set_yticks(np.arange(0, 1.1, 0.1))  # Generate ticks between -0.1 and 1 with step 0.1
ax3.grid()

# Fourth plot
z_vals = np.linspace(0.01,z_pick,1000)
y_vals = f(z_vals,a_pick_max)

ax4.plot(z_vals, y_vals, color='g', label="a = {}, f(a) = {}".format(round(a_pick_max,9),round(max(f(z_vals,a_pick_max)),9)),linewidth=5)
ax4.set_title("Intensity of the field U at value a = {}".format(round(a_pick_max,9)))
ax4.set_xlabel("z (microns)")
ax4.set_ylabel("Intensity")
ax4.ticklabel_format(useOffset=False)
ax4.set_yticks(np.arange(-0.1, 1.1, 0.1))  # Generate ticks between -0.1 and 1 with step 0.1
ax4.legend(loc="upper right")
ax4.grid()

plt.savefig('img/intensity_plots.png')
plt.show()


