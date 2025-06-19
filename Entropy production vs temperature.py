# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:31:34 2025

@author: albri
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
eps0 = 0.005        # base magnon energy (eV)
A0 = 0.005          # coupling strength (eV)
w0 = 0.001          # phonon energy base (eV)
d = 2e-5            # infinitesimal (broadening)
I = 1j
kB = 8.6e-5         # Boltzmann constant (eV/K)
R = 1e-4            # Lorentzian width

# Energy axis
x_vals = np.linspace(-0.01, 0.01, 300)
dx = x_vals[1] - x_vals[0]

# ω axis (not directly used but can be reused later)
omega = np.linspace(0, 0.075, 1000)

# Function definitions
def A_k(k):
    return A0 * np.abs(k)

def w_k(k):
    return w0 * np.abs(k)

def bose(x, T):
    with np.errstate(over='ignore', divide='ignore'):
        return 1 / (np.exp(x / (kB * T)) - 1)

def lorentzian(x, center, width):
    return (width / 2) / (np.pi * ((x - center)**2 + (width / 2)**2))

def G_r(omega, k, eps):
    A = A_k(k)
    wk = w_k(k)
    denom_correction = (2 * wk * A**2) / ((omega - I*d)**2 - wk**2)
    return 1 / (omega - eps - denom_correction)

def G_a(omega, k, eps):
    A = A_k(k)
    wk = w_k(k)
    denom_correction = (2 * wk * A**2) / ((omega + I*d)**2 - wk**2)
    return 1 / (omega - eps - denom_correction)

# Range of temperatures to evaluate
T_vals = np.linspace(50, 300, 300)  # Can adjust resolution

# Range of epsilon values
eps_vals = np.linspace(0.0001, 0.02, 300)  # Avoid zero

# Range of k values
k_vals = range(1, 100)  # Avoid k=0

P_vals = []

for T in T_vals:
    Sp_vals = []
    for eps in eps_vals:
        total_Sp = 0
        for k in k_vals:
            A = A_k(k)
            W = w_k(k)
            n = bose(x_vals, T)
            S = -2 * I * n * (A**2) * (lorentzian(x_vals, W, R) - lorentzian(x_vals, -W, R))
            L = (1/(2*np.pi)) * ((-I * n * 2 * eps) / ((x_vals + I*d)**2 - eps**2)) * (-2) * np.imag(G_r(x_vals, k, eps)) * S * G_a(x_vals, k, eps)
            total_Sp += W * A * np.trapezoid(L, x_vals)
        total_Sp = -2 * np.imag(I * total_Sp) / len(k_vals)
        Sp_vals.append(total_Sp)

    # Now integrate Sp(ε) over ε
    P_T = np.trapezoid(Sp_vals, eps_vals)
    P_vals.append(P_T)

# Plot entropy production vs temperature
plt.figure(figsize=(8, 5))
plt.plot(T_vals, P_vals, label='Integrated Entropy Production vs T')
plt.xlabel('Temperature (K)')
plt.ylabel('Integrated Entropy Production (arb. units)')
plt.title('Total Entropy Production vs Temperature')
plt.grid(True)
plt.legend()
plt.show()
