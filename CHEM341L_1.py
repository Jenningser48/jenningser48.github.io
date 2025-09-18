# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Input data (paste in CSV format)
# ----------------------------
Xwater = np.array([1,0.967113236,0.930676408,0.892103707,0.844574979,0.793206583,0.734766265,0.596849386,0.589895678,0.498609171,0.389980049,0.258547459,0.096288998,0])
Vm_soln = np.array([18.06948847,19.20478977,20.43197392,21.74890387,23.37586147,25.21054072,27.3571281,32.08046307,32.94255296,36.58834002,40.93218234,46.47951207,53.47328198,57.77276147])

X2 = 1 - Xwater   # mole fraction of solute (acetone/ethanol)

# ----------------------------
# R² function 
# ----------------------------
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# ----------------------------
# Find lowest order polynomial with R² ~ 1
# ----------------------------
best_degree = None
best_r2 = -np.inf
best_poly = None

for degree in range(1, 7):  # test 1st–6th order
    coefs = np.polyfit(Xwater, Vm_soln, degree)
    poly = np.poly1d(coefs)
    y_pred = poly(Xwater)
    r2 = r2_score(Vm_soln, y_pred)
    if r2 > 0.999:  # close enough to 1
        best_degree = degree
        best_r2 = r2
        best_poly = poly
        break

print(f"Best polynomial degree: {best_degree}, R² = {best_r2:.5f}")

# Build polynomial equation string
poly_terms = [f"{coef:.6g}*x^{i}" for i, coef in enumerate(best_poly.coefficients[::-1])]
poly_equation = " + ".join(poly_terms).replace("x^0", "").replace("x^1", "x")
print("Polynomial fit equation (Vm vs Xwater):")
print("Vm(x) =", poly_equation)

# ----------------------------
# Derivatives
# ----------------------------
dpoly_water = np.polyder(best_poly)   # derivative wrt Xwater
dVm_dXwater = dpoly_water(Xwater)

# For X2 derivative, fit separately vs X2
coefs2 = np.polyfit(X2, Vm_soln, best_degree)
poly2 = np.poly1d(coefs2)
dpoly2 = np.polyder(poly2)
dVm_dX2 = dpoly2(X2)

# ----------------------------
# Partial molar volumes
# ----------------------------
Vbar_water = Vm_soln - X2 * dVm_dX2
Vbar_2 = Vm_soln - Xwater * dVm_dXwater

# ----------------------------
# Create results DataFrame
# ----------------------------
results = pd.DataFrame({
    "Xwater": Xwater,
    "X2": X2,
    "Vm_soln": Vm_soln,
    "dVm_dXwater": dVm_dXwater,
    "dVm_dX2": dVm_dX2,
    "Vbar_water": Vbar_water,
    "Vbar_2": Vbar_2
})

# ----------------------------
# Print CSV text (copy–paste this into a file)
# ----------------------------
print("\n--- CSV OUTPUT START ---")
print(f"# Polynomial fit (degree {best_degree}, R²={best_r2:.5f})")
print(f"# Vm(x) = {poly_equation}\n")
print(results.to_csv(index=False))
print("--- CSV OUTPUT END ---")

# ----------------------------
# Plot graphs
# ----------------------------

# Graph 3: dVm/dXwater vs Xwater
plt.figure(figsize=(10,6))
plt.plot(Xwater, dVm_dXwater, marker='o')
plt.xlabel("Mole fraction of water, $X_{water}$")
plt.ylabel("dVm/dXwater (cm³/mol)")
plt.title("Graph 3: Derivative wrt $X_{water}$")
plt.grid(True)
plt.show()

# Graph 4: dVm/dX2 vs X2
plt.figure(figsize=(10,6))
plt.plot(X2, dVm_dX2, marker='o', color="green")
plt.xlabel("Mole fraction of component 2, $X_{2}$")
plt.ylabel("dVm/dX2 (cm³/mol)")
plt.title("Graph 4: Derivative wrt $X_{2}$")
plt.grid(True)
plt.show()

# Graph 5: Vbar_water vs Xwater
plt.figure(figsize=(10,6))
plt.plot(Xwater, Vbar_water, marker='s', label="V̄_water")
plt.xlabel("Mole fraction of water, $X_{water}$")
plt.ylabel("Partial molar volume of water (cm³/mol)")
plt.title("Graph 5: V̄_water vs $X_{water}$")
plt.legend()
plt.grid(True)
plt.show()

# Graph 6: Vbar_2 vs Xwater
plt.figure(figsize=(10,6))
plt.plot(Xwater, Vbar_2, marker='s', color="red", label="V̄_2")
plt.xlabel("Mole fraction of water, $X_{water}$")
plt.ylabel("Partial molar volume of component 2 (cm³/mol)")
plt.title("Graph 6: V̄_2 vs $X_{water}$")
plt.legend()
plt.grid(True)
plt.show()


# %%
# ----------------------------
# Print polynomial, derivative, and CSV
# ----------------------------

# Polynomial equation (Vm vs Xwater)
poly_terms = [f"{coef:.6g}*x^{i}" for i, coef in enumerate(best_poly.coefficients[::-1])]
poly_equation = " + ".join(poly_terms).replace("x^0", "").replace("x^1", "x")
print("Polynomial fit equation (Vm vs Xwater):")
print("Vm(x) =", poly_equation)

# Derivative equation (dVm/dXwater)
dpoly_water = np.polyder(best_poly)
dpoly_terms = [f"{coef:.6g}*x^{i}" for i, coef in enumerate(dpoly_water.coefficients[::-1])]
dpoly_equation = " + ".join(dpoly_terms).replace("x^0", "").replace("x^1", "x")
print("\nDerivative equation (dVm/dXwater):")
print("Vm'(x) =", dpoly_equation)

# CSV output
print("\n--- CSV OUTPUT START ---")
print(f"# Polynomial fit (degree {best_degree}, R²={best_r2:.5f})")
print(f"# Vm(x) = {poly_equation}")
print(f"# Vm'(x) = {dpoly_equation}\n")
print(results.to_csv(index=False))
print("--- CSV OUTPUT END ---")


# %%
# ----------------------------
# Print derivative equations in LaTeX format
# ----------------------------

# Derivative wrt Xwater (EtOH is other component)
dpoly_water = np.polyder(best_poly)
dpoly_terms = [f"{coef:.6g} x^{{{i}}}" for i, coef in enumerate(dpoly_water.coefficients[::-1])]
dpoly_equation = " + ".join(dpoly_terms).replace("x^{0}", "").replace("x^{1}", "x")

# Derivative wrt XEtOH (water is other component)
dpoly_2 = np.polyder(poly2)
dpoly2_terms = [f"{coef:.6g} x^{{{i}}}" for i, coef in enumerate(dpoly_2.coefficients[::-1])]
dpoly2_equation = " + ".join(dpoly2_terms).replace("x^{0}", "").replace("x^{1}", "x")

# Print LaTeX code blocks
print("LaTeX for derivative wrt water:")
print(r"\[ \left(\frac{\partial V_m}{\partial X_{\text{water}}}\right)_{T,P,n,\text{EtOH}} = " + dpoly_equation + r"\]")

print("\nLaTeX for derivative wrt EtOH:")
print(r"\[ \left(\frac{\partial V_m}{\partial X_{\text{EtOH}}}\right)_{T,P,n,\text{water}} = " + dpoly2_equation + r"\]")



