

"""
Created on Thu May  8 17:44:28 2025

@author: albri
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
eps0 = 0.005       # base magnon energy (eV)
A0 = 0.005         # coupling strength (eV)
w0 = 0.001         # phonon energy base (eV)
delta = 1e-4       # infinitesimal (broadening)
I = 1j

# ω axis
omega = np.linspace(0, 0.075, 1000)

# Function definitions

def A_k(k):
    return A0 * np.abs(k)

def w_k(k):
    return w0 * np.abs(k)

# Green's function Gr_k(ω)
def G_r(omega, k):
    eps = eps0
    A = A_k(k)
    wk = w_k(k)
    denom_correction = (2 * wk * A**2) / ((omega - I*delta)**2 - wk**2)
    return 1 / (omega - eps - denom_correction)

# Plot for multiple k-values
k_vals = range(0, 10)
plt.figure(figsize=(10, 6))

for k in k_vals:
    G = G_r(omega, k)
    plt.plot(omega, np.imag(G), label=f'k = {round(k, 2)}')

plt.xlabel('ω (eV)')
plt.ylabel('Im[Gᵣₖ(ω)]')
plt.title('Imaginary Part of Retarded Green\'s Function Gᵣₖ(ω)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()