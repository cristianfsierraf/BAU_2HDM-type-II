import matplotlib.pyplot as plt
import numpy as np

#All the constants and definitions from the WKB method are taken from PHYS. REV. D 101, 063525 (2020)

#-------------#
#  Constants  #
#-------------#

GF = 1.16638e-5 #Fermi constant
v = 1/np.sqrt(np.sqrt(2)*GF) #VEV at zero temperature
g2 = 0.652  #weak coupling
mt = 173.34 #top quark mass
yt = np.sqrt(2)*mt/v  #top Yukawa coupling
YBobs = 8.59*(10**-11) #Planck collaboration measurement

#Factors involved in the WKB transport equations

D0 = 1. #D0 for fermions
D0h = 2. #D0 for bosons

#k factors related to the chemical potential defined as in Eq 72 of hep-ph/0412354

kQ30 = 6. # quark doublets
kT0 = 3. #right-handed top quark
kB0 = 3. #right-handed bottom quark
kH0 = 2. #Higgs boson

#auxiliary combinations of the k factors involved in the solution of the diffusion equation from hep-ph/0412354

a = kH0*(9*kQ30 + 9*kT0 + kB0)
b = 9*kQ30*kT0 + kQ30*kB0 + 4*kT0*kB0
r1 = (9*kQ30*kT0 - 5*kQ30*kB0 - 8*kT0*kB0)/a
r2 = kH0*kB0**2*(5*kQ30 + 4*kT0)*(kQ30 + 2*kT0)/(a**2)

# Testing fiducial values for various parameters entering in the difussion equation and the profile functions

Tn = 100. # Nucleation temperature
vn = 100. # Nucleation VEV
Lambda = 1000. #Scale for the testing profile functions
lw = 5./Tn # Bubble wall width 
vw_fixed = 0.1 # Bubble wall velocity
T_fixed = Tn 

#-----------------------------#
# Test kink profile functions #
#-----------------------------#

#TO DO
#These functions should be replaced by an analytical approximation to the actual field profiles obtained by the other code

#phase
def theta(z):
  return np.arctan(vn*(1 - np.tanh(z/lw))/Lambda)
#Higgs profile
def phi(z):
  return vn*(1 + np.tanh(z/lw))/2
#Top quark square-mass profile. This profile always enters the calculation as a square
def m2t(z):
  return (1/2)*(yt**2)*(phi(z)**2) 

#---------------------------------------------#
# Definitions for source term in WKB approach #
#---------------------------------------------#
#Relativistic factor
def gamma(vw):
  return 1/np.sqrt(1-vw**2)

def ReQ81(vw,T):
  return 3*(-1+vw**2)*(-np.log(np.absolute(1-vw))+np.log(1+vw))/(4*(np.pi*T)**2)

def ImQ81(vw,T):
  return -3*(-1+vw**2)*np.pi*1/(4*(np.pi*T)**2)

def Q81(vw,T):
  return ReQ81(vw,T)+ImQ81(vw,T)*1j

def Q82(vw,T):
  return 3*np.arctanh(vw)/(2*vw*(np.pi*T*gamma(vw))**2)

#Diffusion functions for quarks and Higgs from PhysRevD.104.083507
def Dq(T):
  return 7.2/T

def Dh(T):
  return 20/T

#Bubble wall velocity-dependent functions associated for fermions and bosons:

#fermions
def D2(vw):
  return (vw*(-1+2*vw**2) + ((-1+vw**2)**2)*np.arctanh(vw))/vw**3
#bosons
def D2h(vw):
  return 2*D2(vw)

#Total thermal decay witdh
def Gammatot(dalpha,d0,d2):
  return d2/(d0*dalpha)

# Diffusion coefficient in WKB
def DWKB(vw,dalpha,d0,d2):
  return (d2-vw**2*d0)/(d0*Gammatot(dalpha,d0,d2))

#Thermal decay widths in WKB formalism from PhysRevD.104.083507

def Gamma_mt(z,T):
   return 2*kT0*0.79*m2t(z)/T

def Gamma_h(z,T):
   return kH0*0.79*(g2**phi(z))**2/(4*25*T)

def Gamma_ss(T):
   return kT0*(8.7*10**-3)*T

def Gamma_ws(T):
   return 2*kT0*(6.3*10**-6)*T

#Effective thermal decay width

def Gammabar(z,T):
   return a*(Gamma_mt(z,T)+Gamma_h(z,T))/(kH0*(a+b))

#Effective diffusion coefficient

def Dbar(vw,T):
  return (1/(a+b))*(b*(DWKB(vw,Dq(T),D0,D2(vw)))+a*(DWKB(vw,Dh(T),D0h,D2h(vw))))

#Kappa factors in the computation of the amplitude of the Higgs field solution

def kappa_p(z,vw,T):
  return (vw + np.sqrt(vw**2 + 4*Gammabar(z,T)*Dbar(vw,T)))/(2*Dbar(vw,T))

def kappa_m(z,vw,T):
  return (vw - np.sqrt(vw**2 + 4*Gammabar(z,T)*Dbar(vw,T)))/(2*Dbar(vw,T))

#Alpha factors in the computation of the baryon asymmetry YB (called "lambda" in paper with Michael)

def alpha_p(vw,T):
  return (vw + np.sqrt(4*DWKB(vw,Dq(T),D0,D2(vw))*Gamma_ws(T)*(15./4)+ vw**2 ))/(2*DWKB(vw,Dq(T),D0,D2(vw)))

def alpha_m(vw,T):
  return (vw - np.sqrt(4*DWKB(vw,Dq(T),D0,D2(vw))*Gamma_ws(T)*(15./4)+ vw**2 ))/(2*DWKB(vw,Dq(T),D0,D2(vw)))

#Density entropy in the BAU calculation
def density_entropy(T):
    gstar2HDM = 110.75 # degress of freedom in the 2HDM
    return (2*np.pi**2/45.)*gstar2HDM*T**3

#---------------------#
# Auxiliary functions #
#---------------------#

# Derivative of theta
def theta_derivative_function(z):
    theta_vals = theta(z)  # Calculate theta values for the given z
    return np.gradient(theta_vals, z)  # Calculate and return the gradient

# Derivative of the product of m2t(z) and the derivative of theta(z)
def product_derivative(z):
    mtop_2 = m2t(z)  # Calculate m2t values
    theta_derivative = theta_derivative_function(z)  # Get the derivative of theta
    product = mtop_2 * theta_derivative  # Calculate the product
    return np.gradient(product, z)  # Return the derivative of the product

# Auxiliary function for numerical integration
def integral(func, vw, T, limit_a, limit_b):
    z_vals = np.linspace(limit_a, limit_b, 30000)  
    func_vals = func(z_vals, vw, T)  
    return np.trapz(func_vals, z_vals) 

# Auxiliary function for numerical integration
def Integral_YB(func, vw, T, Lw, limit_a, limit_b):
    z_vals = np.linspace(limit_a, limit_b, 30000)  
    func_vals = func(z_vals, vw, T, Lw)  
    return np.trapz(func_vals, z_vals) 
