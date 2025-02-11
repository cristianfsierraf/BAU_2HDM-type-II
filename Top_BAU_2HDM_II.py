import matplotlib.pyplot as plt
import numpy as np
from Definitions import *

# Definition of the SWKB CP-violating source from PHYS. REV. D 101, 063525 (2020), the Q9 part is negligible
def SWKB_function(z, vw, T):
    S1 = product_derivative(z) * vw * gamma(vw) * Q81(vw, T)
    S2 = product_derivative(z) * vw * gamma(vw) * Q82(vw, T)
    return (a / (a + b)) * kT0 * ((T**2) / 6) * (S1 / D0 - (vw * np.gradient(S1, z) + np.gradient(S2, z)) / (D0 * Gammatot(Dq(T), D0, D2(vw))))

# Plotting the SWKB function:
# Define the range for z
z_vals = np.linspace(-0.2, 0.2, 100)

# Calculate y-values using SWKB_function (debugging)
swkb_vals = np.abs(SWKB_function(z_vals, vw_fixed, T_fixed))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(z_vals, swkb_vals, label=r'$\upsilon_{w}=0.1,\quad\,T_{n}=100\,\mathrm{Gev}$', color='red')
plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.axvline(0, color='grey', lw=0.5, ls='--')
plt.xlabel(r'$z$',fontsize=14)
plt.ylabel(r'$|S_{WKB}|$',fontsize=14)
plt.legend()
plt.grid()
plt.show()

# Integrand of amplitude A = A1 + A2 for H(z) = A exp(vw z / Dbar)
def A1(z, vw, T):
    return SWKB_function(z, -vw, T) * np.exp(-kappa_p(z, vw, T) * z) / (Dbar(vw, T) * kappa_p(z, vw, T))

def A2(z, vw, T):
    return SWKB_function(z, -vw, T) * ((kappa_m(z, vw, T) / (vw * kappa_p(z, vw, T))) + np.exp(-vw * z / Dbar(vw, T)) / vw)

# Plot real and imaginary parts (just for debugging purposes)
#A1_plot_re = np.real(A1(z_vals, vw_fixed, T_fixed))
#A1_plot_im = np.imag(A1(z_vals, vw_fixed, T_fixed))
#A2_plot_re = np.real(A2(z_vals, vw_fixed, T_fixed))
#A2_plot_im = np.imag(A2(z_vals, vw_fixed, T_fixed))

#fig = plt.figure(figsize = (10, 5))
# Create the plot
#plt.title(r'$\upsilon_{w}=0.1,\quad\,T_{n}=100\,\mathrm{GeV}$')
#plt.plot(z_vals, A1_plot_re, label=r'$\mathrm{Re}(A_1)(z)$', color='blue')
#plt.plot(z_vals, A1_plot_im, label=r'$\mathrm{Im}(A_1)(z)$', color='orange')
#plt.plot(z_vals, A2_plot_re, label=r'$\mathrm{Re}(A_2)(z)$', color='green')
#plt.plot(z_vals, A2_plot_im, label=r'$\mathrm{Im}(A_2)(z)$', color='red')
#plt.legend()
#plt.grid()
# Show the plot
#plt.show()

#Left-handed particle density
def n_L(z, vw, T, Lw, z_cut_off):
    Amplitude = integral(A1, vw, T, 0, z_cut_off) + integral(A2, vw, T, -Lw/2., 0.)  # Replace z variable cut_off with a finite upper limit as needed
    prefactor = -(r1 + r2*(vw**2/(Gamma_ss(T)*Dbar(vw, T))*(1-(DWKB(vw,Dq(T),D0,D2(vw))/Dbar(vw,T)))))
    Higgs = Amplitude *np.exp(vw*z/(Dbar(vw, T)))
    nL = prefactor*Higgs
    return nL

#print("prefactor = ", -(r1 + r2*(vw_fixed**2/(Gamma_ss(T_fixed)*Dbar(vw_fixed, T_fixed))*(1-(DWKB(vw_fixed,Dq(T_fixed),D0,D2(vw_fixed))/Dbar(vw_fixed,T_fixed))))))

#print("nL(z=0.2) = ",np.real(n_L(0.2, vw_fixed, T_fixed, 0.4, 0.2)))

# Calculate real part (just for debugging purposes) of n_L for each value of z
nL_vals = np.real(n_L(z_vals, vw_fixed, T_fixed, 0.4,0.2))

# Plotting the n_L function
plt.figure(figsize=(10, 5))
plt.plot(z_vals, nL_vals, label='n_L(z, vw=0.1, Tn=100 GeV)', color='blue')
plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.axvline(0, color='grey', lw=0.5, ls='--')
plt.xlabel('z')
plt.ylabel('n_L(z, vw, T, Lw)')
plt.legend()
plt.grid()
plt.show()

#Baryon-asymmetry: Y_B integrand
def Integrand_YB(z, vw, T, Lw):
    z_cut_off= 50.
    return n_L(z, vw, T, Lw, z_cut_off)*np.exp(-alpha_m(vw,T)*z)

#Baryon-asymmetry: YB/YB_Bobs
def BAU(vw, T, Lw):
    cut_off = 50.
    prefactor = -3*Gamma_ws(T)/(2*density_entropy(T)*DWKB(vw,Dq(T),D0,D2(vw))*gamma(vw)*alpha_p(vw,T))
    Integration = Integral_YB(Integrand_YB, vw, T, Lw, -cut_off, -Lw/2.)  # Replace cut_off with a finite upper limit as needed
    YB = prefactor*Integration
    BAU = YB/YBobs
    return BAU

vw_vals = np.linspace(0.01,1.0,100)

plot_BAU = [np.abs(BAU(vw, T_fixed, lw)) for vw in vw_vals]

fig = plt.figure(figsize = (10, 5))
# Create the plot
plt.plot(vw_vals, plot_BAU,label=r'$L_{w}=5/T_{n},\quad\,T_{n}=100\,\mathrm{GeV}$')
plt.xlabel(r'$\upsilon_{w}$',fontsize=14)
plt.ylabel('|BAU|',fontsize=12)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.axhline(y = 1, color = 'r', linestyle = '-')
plt.legend()
plt.grid()
plt.show()
