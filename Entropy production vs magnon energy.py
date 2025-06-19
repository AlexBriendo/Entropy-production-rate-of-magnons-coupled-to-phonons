# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
eps0 = 0.005       # base magnon energy (eV)
A0 = 0.005       # coupling strength (eV)
w0 = 0.001         # phonon energy base (eV)
d = 1e-4       # infinitesimal (broadening)
I = 1j
B = 1        # k*T (eV) mettre boltzmann ici et utiliser temp en paramètre pr plot (peut aussi plot en fonction de A)
R = 1e-4       # Lorentzian width


# ω axis
omega = np.linspace(0, 0.075, 1000)

# Function definitions

def A_k(k):
    return A0 * np.abs(k)

def w_k(k):
    return w0 * np.abs(k)


# Bose-Einstein distribution
def bose(x):
    with np.errstate(over='ignore', divide='ignore'):
        return 1 / (np.exp(x / B) - 1)

# Lorentzian function
def lorentzian(x, center, width):
    return (width / 2) / (np.pi * ((x - center)**2 + (width / 2)**2))

# Energy axis
x_vals = np.linspace(-0.01, 0.01, 2000)
dx = x_vals[1] - x_vals[0]

# Green's function Gr_k(ω)
def G_r(omega, k):
    eps = eps0
    A = A_k(k)
    wk = w_k(k)
    denom_correction = (2 * wk * A**2) / ((omega - I*d)**2 - wk**2)
    return 1 / (omega - eps - denom_correction)

def G_a(omega, k):
    eps = eps0
    A = A_k(k)
    wk = w_k(k)
    denom_correction1 = (2 * wk * A**2) / ((omega + I*d)**2 - wk**2)
    return 1 / (omega - eps - denom_correction1)

# Plot for multiple k-values
k_vals = range(0, 100)

# Varying magnon energy
eps_vals = np.linspace(0 , 0.02 , 100)
Sp_vals = []

for eps in eps_vals:
    total_Sp = 0
    for k in k_vals:
        A = A_k(k)
        W = w_k(k)
        n = bose(x_vals)
        S = -2 * I * n * (A**2) * (lorentzian(x_vals, W, R) - lorentzian(x_vals, -W, R))
        L = (1/(2*np.pi))*((-I*n*2*eps)/((x_vals+I*d)**2-eps**2))*(-2)*np.imag(G_r(x_vals, k)) * S * G_a(x_vals, k)
        total_Sp += W*A*np.trapezoid(L, x_vals)  # integrate over x
    total_Sp = -2 * np.imag(I * total_Sp) / len(k_vals)  # normalize over k
    Sp_vals.append(total_Sp)

plt.figure(figsize=(8, 5))
plt.plot(eps_vals, Sp_vals, label='Entropy Production rate vs ε (summed over k)')
plt.xlabel('Magnon Energy ε (eV)')
plt.ylabel('Entropy Production (arb. units)')
plt.title('Entropy Production vs Magnon Energy with k-sum')
plt.grid(True)
plt.legend()
plt.show()