import numpy as np
import matplotlib.pyplot as plt

# Constants
eps0 = 0.005        # base magnon energy (eV)
w0 = 0.001          # phonon energy base (eV)
d = 1e-6           # infinitesimal (broadening)
I = 1j
kB = 8.6e-5         # Boltzmann constant (eV/K)
R = 1e-4            # Lorentzian width
T = 300           # Fixed temperature (K)

# Energy axis
x_vals = np.linspace(-0.01, 0.01, 300)
dx = x_vals[1] - x_vals[0]

# Range of ε values
eps_vals = np.linspace(0.0001, 0.02, 300)

# Range of k values
k_vals = range(1, 100)  # Avoid k = 0

# Coupling strength values to evaluate
A0_vals = np.linspace(0, 0.001, 100)  # Range of A0 values
P_vals = []

# Bose-Einstein distribution
def bose(x, T):
    with np.errstate(over='ignore', divide='ignore'):
        return 1 / (np.exp(x / (kB * T)) - 1)

# Lorentzian
def lorentzian(x, center, width):
    return (width / 2) / (np.pi * ((x - center)**2 + (width / 2)**2))

# Main loop over A0 values
for A0 in A0_vals:

    # Define A_k and w_k with local A0
    def A_k(k):
        return A0 * np.abs(k)

    def w_k(k):
        return w0 * np.abs(k)

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

    # Integrate over ε
    P_A0 = np.trapezoid(Sp_vals, eps_vals)
    P_vals.append(P_A0)
    
F=np.trapezoid(P_vals, A0_vals)

# Plot entropy production vs coupling strength A0
plt.figure(figsize=(8, 5))
plt.plot(A0_vals, P_vals, label='Entropy Production vs Coupling Strength A₀')
plt.xlabel('Coupling Strength A₀ (eV)')
plt.ylabel('Integrated Entropy Production (arb. units)')
plt.title('Total Entropy Production vs Coupling Strength (T = 3000 K)')
plt.grid(True)
plt.legend()
plt.show()
