---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: qutip-dev
  language: python
  name: python3
---

# HEOM 5a: Fermionic single impurity model

+++

## Introduction

Here we model a single fermion coupled to two electronic leads or reservoirs (e.g.,  this can describe a single quantum dot, a molecular transistor, etc).  Note that in this implementation we primarily follow the definitions used by Christian Schinabeck in his dissertation https://opus4.kobv.de/opus4-fau/files/10984/DissertationChristianSchinabeck.pdf and related publications.

Notation:

* $K=L/R$ refers to  left or right leads.
* $\sigma=\pm$ refers to input/output

We choose a Lorentzian spectral density for the leads, with a peak at the chemical potential. The latter simplifies a little the notation required for the correlation functions, but can be relaxed if neccessary.

$$J(\omega) = \frac{\Gamma  W^2}{((\omega-\mu_K)^2 +W^2 )}$$

The Fermi distribution function is:

$$f_F (x) = (\exp(x) + 1)^{-1}$$

Together these allow the correlation functions to be expressed as:

$$C^{\sigma}_K(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega e^{\sigma i \omega t} \Gamma_K(\omega) f_F[\sigma\beta(\omega - \mu)]$$

As with the bosonic case we can expand these in an exponential series using Matsubara, Pade, or fitting approaches.

The Pade decomposition approximates the Fermi distubition as

$$f_F(x) \approx f_F^{\mathrm{approx}}(x) = \frac{1}{2} - \sum_l^{l_{max}} \frac{2k_l x}{x^2 + \epsilon_l^2}$$

where $k_l$ and $\epsilon_l$ are co-efficients defined in J. Chem Phys 133,10106.

Evaluating the integral for the correlation functions gives,

$$C_K^{\sigma}(t) \approx \sum_{l=0}^{l_{max}} \eta_K^{\sigma_l} e^{-\gamma_{K,\sigma,l}t}$$

where:

$$\eta_{K,0} = \frac{\Gamma_KW_K}{2} f_F^{approx}(i\beta_K W)$$

$$\gamma_{K,\sigma,0} = W_K - \sigma i\mu_K$$ 

$$\eta_{K,l\neq 0} = -i\cdot \frac{k_m}{\beta_K} \cdot \frac{\Gamma_K W_K^2}{-\frac{\epsilon^2_m}{\beta_K^2} + W_K^2}$$

$$\gamma_{K,\sigma,l\neq 0}= \frac{\epsilon_m}{\beta_K} - \sigma i \mu_K$$

In this notebook we:

* compare the Matsubara and Pade approximations and contrast them with the analytical result for the current between the system and the leads.

* plot the current through the qubit as a function of the different between the voltages of the leads.

+++

## Setup

```{code-cell} ipython3
import contextlib
import dataclasses
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import qutip
from qutip import (
    basis,
    destroy,
    expect,
)
from qutip.solver.heom import (
    HEOMSolver,
    LorentzianBath,
    LorentzianPadeBath,
)

from ipywidgets import IntProgress
from IPython.display import display

%matplotlib inline
```

## Helpers

```{code-cell} ipython3
@contextlib.contextmanager
def timer(label):
    """ Simple utility for timing functions:

        with timer("name"):
            ... code to time ...
    """
    start = time.time()
    yield
    end = time.time()
    print(f"{label}: {end - start}")
```

```{code-cell} ipython3
# Solver options:

# We set store_ados to True so that we can
# use the auxilliary density operators (ADOs)
# to calculate the current between the leads
# and the system.

options = {
    "nsteps": 1500,
    "store_states": True,
    "store_ados": True,
    "rtol": 1e-12,
    "atol": 1e-12,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## System and bath definition

And let us set up the system Hamiltonian, bath and system measurement operators:

```{code-cell} ipython3
# Define the system Hamiltonian:

# The system is a single fermion with energy level split e1:
d1 = destroy(2)
e1 = 1.0
H = e1 * d1.dag() * d1
```

```{code-cell} ipython3

from qutip.core.environment import LorentzianEnvironment


envL=LorentzianEnvironment(T= 0.025851991,W=1,mu=1,gamma=0.01,Nk=20)
envR=LorentzianEnvironment(T= 0.025851991,W=1,mu=-1,gamma=0.01,Nk=20)
```

## Spectral density

Let's plot the spectral density.

```{code-cell} ipython3
w_list = np.linspace(-30, 30, 2000)

fig, ax = plt.subplots(figsize=(12, 7))

spec_L = envL.spectral_density(w_list)
spec_R = envR.spectral_density(w_list)

ax.plot(
    w_list, spec_L,
    "b--", linewidth=3,
    label=r"J_L(w)",
)
ax.plot(
    w_list, spec_R,
    "r--", linewidth=3,
    label=r"J_R(w)",
)

ax.set_xlabel("w")
ax.set_ylabel(r"$J(\omega)$")
ax.legend();
```

```{code-cell} ipython3
from qutip.core.environment import FermionicEnvironment
```

```{code-cell} ipython3
# fenvL=FermionicEnvironment.from_spectral_density(envL.spectral_density(w_list),w_list,T=0.025851991,mu=1)
# fenvR=FermionicEnvironment.from_spectral_density(envR.spectral_density(w_list),w_list,T=0.025851991,mu=-1)
# fenvL=FermionicEnvironment.from_power_spectra(envL.power_spectrum_minus(w_list),w_list,T=0.025851991,mu=1,sigma=-1)
# fenvR=FermionicEnvironment.from_power_spectra(envR.power_spectrum_minus(w_list),w_list,T=0.025851991,mu=-1,sigma=-1)
tk=np.linspace(0,120,1000)
fenvL=FermionicEnvironment.from_correlation_functions(envL.correlation_function_plus(tk),tk,T=0.025851991,mu=1,sigma=1)
fenvR=FermionicEnvironment.from_correlation_functions(envR.correlation_function_plus(tk),tk,T=0.025851991,mu=-1,sigma=1)
```

```{code-cell} ipython3

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(
    w_list,envR.spectral_density(w_list) ,
    "r", linewidth=3,
    label=r"J_R(w)",
)
ax.plot(
    w_list, fenvR.spectral_density(w_list),
    "g--", linewidth=3,
    label=r"J_R_env(w)",
)
ax.set_xlabel("w")
ax.set_ylabel(r"$J(\omega)$")
ax.legend();
```

```{code-cell} ipython3
fenvR.spectral_density(w_list)[1200]
```

## Emission and absorption by the leads

Next let's plot the emission and absorption by the leads.

```{code-cell} ipython3

fig, ax = plt.subplots(figsize=(12, 7))

# Left lead emission and absorption

gam_L_in = envL.power_spectrum_plus(w_list)
gam_L_out = envL.power_spectrum_minus(w_list)

ax.plot(
    w_list, gam_L_in,
    "b--", linewidth=3,
    label=r"S_L(w) input (absorption)",
)
ax.plot(
    w_list, gam_L_out,
    "r--", linewidth=3,
    label=r"S_L(w) output (emission)",
)

# Right lead emission and absorption

gam_R_in = envR.power_spectrum_plus(w_list)
gam_R_out = envR.power_spectrum_minus(w_list)

ax.plot(
    w_list, gam_R_in,
    "b", linewidth=3,
    label=r"S_R(w) input (absorption)",
)
ax.plot(
    w_list, gam_R_out,
    "r", linewidth=3,
    label=r"S_R(w) output (emission)",
)

ax.set_xlabel("w")
ax.set_ylabel(r"$S(\omega)$")
ax.legend();
```

```{code-cell} ipython3
w_list = np.linspace(-20, 20, 1600)

fig, ax = plt.subplots(figsize=(12, 7))

# Left lead emission and absorption

gam_L_in = envL.power_spectrum_plus(w_list)
gam_L_out = envL.power_spectrum_minus(w_list)

# ax.plot(
#     w_list, gam_L_in,
#     "b", linewidth=3,
#     label=r"S_L(w) input (absorption)",
# )
ax.plot(
    w_list, gam_L_out,
    "r", linewidth=3,
    label=r"S_L(w) output (emission)",
)

# Right lead emission and absorption

gam_R_in = fenvL.power_spectrum_plus(w_list)
gam_R_out = fenvL.power_spectrum_minus(w_list)

# ax.plot(
#     w_list, gam_R_in,
#     "r--", linewidth=3,
#     label=r"S_R(w) input (absorption)",
# )
# ax.plot(
#     w_list, gam_R_out,
#     "b--", linewidth=3,
#     label=r"S_R(w) output (emission)",
# )
# ax.axvline(x=1)
# ax.set_ylim(0,0.04)
ax.set_xlabel("w")
ax.set_ylabel(r"$S(\omega)$")
ax.legend();
```

```{code-cell} ipython3
from qutip.utilities import fermi_dirac
f = fermi_dirac(w_list, 1/envL.T, envL.mu)
ff=1/f
print(ff)
ff -= 1
result = fenvL.power_spectrum_plus(w_list)
```

```{code-cell} ipython3
np.argmax(gam_R_out)
```

```{code-cell} ipython3
w_list[1572]
```

```{code-cell} ipython3
plt.plot(w_list,envL.power_spectrum_plus(w_list)-fenvL.power_spectrum_plus(w_list))
```

```{code-cell} ipython3
fenvL.power_spectrum_plus(w_list[1572])
```

```{code-cell} ipython3
assert 1==0
```

```{code-cell} ipython3
tk = np.linspace(-60, 60, 1000)

fig, ax = plt.subplots(figsize=(12, 7))

# Left lead emission and absorption

gam_L_in = envL.correlation_function_plus(tk) .imag
gam_L_out = envL.correlation_function_minus(tk)

ax.plot(
    tk, gam_L_in,
    "b", linewidth=3,
    label=r"S_L(w) input (absorption)",
)
# ax.plot(
#     w_list, gam_L_out,
#     "r", linewidth=3,
#     label=r"S_L(w) output (emission)",
# )

# Right lead emission and absorption

gam_R_in = fenvL.correlation_function_plus(tk).imag
# gam_R_out = fenvL.correlation_function_minus(w_list)

ax.plot(
    tk, gam_R_in,
    "r--", linewidth=3,
    label=r"S_R(w) input (absorption)",
)
# ax.plot(
#     w_list, gam_R_out,
#     "b--", linewidth=3,
#     label=r"S_R(w) output (emission)",
# )

ax.set_xlabel("w")
ax.set_ylabel(r"$S(\omega)$")
ax.legend();
```

```{code-cell} ipython3
w_list = np.linspace(0, 60, 1000)

fig, ax = plt.subplots(figsize=(12, 7))

# Left lead emission and absorption

gam_L_in = envL.correlation_function_plus(w_list).imag 
gam_L_out = envL.correlation_function_minus(w_list).imag

ax.plot(
    w_list, gam_L_in,
    "b", linewidth=3,
    label=r"S_L(w) input (absorption)",
)
# ax.plot(
#     w_list, gam_L_out,
#     "r", linewidth=3,
#     label=r"S_L(w) output (emission)",
# )

# Right lead emission and absorption

gam_R_in = fenvL.correlation_function_plus(w_list).imag
#gam_R_out = fenvL.correlation_function_minus(w_list).imag

ax.plot(
    w_list, gam_R_in,
    "r--", linewidth=3,
    label=r"S_R(w) input (absorption)",
)
# ax.plot(
#     w_list, gam_R_out,
#     "b--", linewidth=3,
#     label=r"S_R(w) output (emission)",
# )

ax.set_xlabel("w")
ax.set_ylabel(r"$S(\omega)$")
ax.legend();
```

```{code-cell} ipython3
cm=envL.correlation_function_minus(w_list)
```

```{code-cell} ipython3
cmm=lambda t:fenvL.correlation_function_plus(t-1j*(1/envL.T)).conj()*np.exp(-envL.mu/envL.T)
```

```{code-cell} ipython3
plt.plot(w_list,cm.real)
plt.plot(w_list,cmm(w_list).real*1e17)
```

## Comparing the Matsubara and Pade approximations

Let's start by solving for the evolution using a Pade expansion of the correlation function of the Lorentzian spectral density:

```{code-cell} ipython3
# HEOM dynamics using the Pade approximation:

# Times to solve for and initial system state:
tlist = np.linspace(0, 100, 1000)
rho0 = basis(2, 0) * basis(2, 0).dag()

Nk = 10  # Number of exponents to retain in the expansion of each bath

bathL = LorentzianPadeBath(
    bath_L.Q, bath_L.gamma, bath_L.W, bath_L.mu, bath_L.T,
    Nk, tag="L",
)
bathR = LorentzianPadeBath(
    bath_R.Q, bath_R.gamma, bath_R.W, bath_R.mu, bath_R.T,
    Nk, tag="R",
)

with timer("RHS construction time"):
    solver_pade = HEOMSolver(H, [bathL, bathR], max_depth=2, options=options)

with timer("ODE solver time"):
    result_pade = solver_pade.run(rho0, tlist)

with timer("Steady state solver time"):
    rho_ss_pade, ado_ss_pade = solver_pade.steady_state()
```

Now let us plot the result which shows the decay of the initially excited impurity. This is not very illuminating, but we will compare it with the Matsubara expansion and analytic solution sortly:

```{code-cell} ipython3
# Plot the Pade results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

axes.plot(
    tlist, expect(result_pade.states, rho0),
    'r--', linewidth=2,
    label="P11 (Pade)",
)
axes.axhline(
    expect(rho_ss_pade, rho0),
    color='r', linestyle="dotted", linewidth=1,
    label="P11 (Pade steady state)",
)

axes.set_xlabel('t', fontsize=28)
axes.legend(fontsize=12);
```

Now let us do the same for the Matsubara expansion:

```{code-cell} ipython3
# HEOM dynamics using the Matsubara approximation:

bathL = LorentzianBath(
    bath_L.Q, bath_L.gamma, bath_L.W, bath_L.mu, bath_L.T,
    Nk, tag="L",
)
bathR = LorentzianBath(
    bath_R.Q, bath_R.gamma, bath_R.W, bath_R.mu, bath_R.T,
    Nk, tag="R",
)

with timer("RHS construction time"):
    solver_mats = HEOMSolver(H, [bathL, bathR], max_depth=2, options=options)

with timer("ODE solver time"):
    result_mats = solver_mats.run(rho0, tlist)

with timer("Steady state solver time"):
    rho_ss_mats, ado_ss_mats = solver_mats.steady_state()
```

We see a marked difference in the Matsubara vs Pade results:

```{code-cell} ipython3
# Plot the Pade results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

axes.plot(
    tlist, expect(result_pade.states, rho0),
    'r--', linewidth=2,
    label="P11 (Pade)",
)
axes.axhline(
    expect(rho_ss_pade, rho0),
    color='r', linestyle="dotted", linewidth=1,
    label="P11 (Pade steady state)",
)

axes.plot(
    tlist, expect(result_mats.states, rho0),
    'b--', linewidth=2,
    label="P11 (Mats)",
)
axes.axhline(
    expect(rho_ss_mats, rho0),
    color='b', linestyle="dotted", linewidth=1,
    label="P11 (Mats steady state)",
)

axes.set_xlabel('t', fontsize=28)
axes.legend(fontsize=12);
```

But which is more correct? The Matsubara or the Pade result?

One advantage of this simple model is that the steady state current to the baths is analytically solvable, so we can check convergence of the result by calculating it analytically (the sum of the currents to and from the system in the steady state must be zero, so the current from one bath is the same as the current to the other).

See the [QuTiP-BoFiN paper](https://arxiv.org/abs/2010.10806) for a detailed description and references for the analytic result. Below we just perform the required integration numerically.

```{code-cell} ipython3
def analytical_steady_state_current(bath_L, bath_R, e1):
    """ Calculate the analytical steady state current. """

    def integrand(w):
        return (2 / np.pi) * (
            bath_L.J(w) * bath_R.J(w) * (bath_L.fF(w) - bath_R.fF(w)) /
            (
                (bath_L.J(w) + bath_R.J(w))**2 +
                4*(w - e1 - bath_L.lamshift(w) - bath_R.lamshift(w))**2
            )
        )

    def real_part(x):
        return np.real(integrand(x))

    def imag_part(x):
        return np.imag(integrand(x))

    # in principle the bounds for the integral should be rechecked if
    # bath or system parameters are changed substantially:
    bounds = [-10, 10]

    real_integral, _ = quad(real_part, *bounds)
    imag_integral, _ = quad(imag_part, *bounds)

    return real_integral + 1.0j * imag_integral


curr_ss_analytic = analytical_steady_state_current(bath_L, bath_R, e1)

print(f"Analytical steady state current: {curr_ss_analytic}")
```

To compare the analytical result above with the result from the HEOM, we need to be able to calculate the current from the system to the bath from the HEOM result. In the HEOM description, these currents are captured in the first level auxilliary density operators (ADOs).

In the function `state_current(...)` below, we extract the first level ADOs for the specified bath and sum the contributions to the current from each:

```{code-cell} ipython3
def state_current(ado_state, bath_tag):
    """ Determine current from the given bath (either "R" or "L") to
        the system in the given ADO state.
    """
    level_1_aux = [
        (ado_state.extract(label), ado_state.exps(label)[0])
        for label in ado_state.filter(level=1, tags=[bath_tag])
    ]

    def exp_sign(exp):
        return 1 if exp.type == exp.types["+"] else -1

    def exp_op(exp):
        return exp.Q if exp.type == exp.types["+"] else exp.Q.dag()

    return -1.0j * sum(
        exp_sign(exp) * (exp_op(exp) * aux).tr()
        for aux, exp in level_1_aux
    )
```

Now we can calculate the steady state currents from the Pade and Matsubara HEOM results:

```{code-cell} ipython3
curr_ss_pade_L = state_current(ado_ss_pade, "L")
curr_ss_pade_R = state_current(ado_ss_pade, "R")

print(f"Pade steady state current (L): {curr_ss_pade_L}")
print(f"Pade steady state current (R): {curr_ss_pade_R}")
```

```{code-cell} ipython3
curr_ss_mats_L = state_current(ado_ss_mats, "L")
curr_ss_mats_R = state_current(ado_ss_mats, "R")

print(f"Matsubara steady state current (L): {curr_ss_mats_L}")
print(f"Matsubara steady state current (R): {curr_ss_mats_R}")
```

Note that the currents from each bath balance as is required by the steady state, but the value of the current is different for the Pade and Matsubara results.

Now let's compare all three:

```{code-cell} ipython3
print(f"Pade current (R): {curr_ss_pade_R}")
print(f"Matsubara current (R): {curr_ss_mats_R}")
print(f"Analytical curernt: {curr_ss_analytic}")
```

In this case we observe that the Pade approximation has converged more closely to the analytical current than the Matsubara.

The Matsubara result could be improved by increasing the number of terms retained in the Matsubara expansion (i.e. increasing `Nk`).

+++

## Current as a function of bias voltage

+++

Now lets plot the current as a function of bias voltage (the bias voltage is the parameter `theta` for the two baths).

We will calculate the steady state current for each `theta` both analytically and using the HEOM with the Pade correlation expansion approximation.

```{code-cell} ipython3
# Theta (bias voltages)

thetas = np.linspace(-4, 4, 100)

# Setup a progress bar:

progress = IntProgress(min=0, max=2 * len(thetas))
display(progress)

# Calculate the current for the list of thetas


def current_analytic_for_theta(e1, bath_L, bath_R, theta):
    """ Return the analytic current for a given theta. """
    current = analytical_steady_state_current(
        bath_L.replace(theta=theta),
        bath_R.replace(theta=theta),
        e1,
    )
    progress.value += 1
    return np.real(current)


def current_pade_for_theta(H, bath_L, bath_R, theta, Nk):
    """ Return the steady state current using the Pade approximation. """
    bath_L = bath_L.replace(theta=theta)
    bath_R = bath_R.replace(theta=theta)

    bathL = LorentzianPadeBath(
        bath_L.Q, bath_L.gamma, bath_L.W, bath_L.mu, bath_L.T,
        Nk, tag="L",
    )
    bathR = LorentzianPadeBath(
        bath_R.Q, bath_R.gamma, bath_R.W, bath_R.mu, bath_R.T,
        Nk, tag="R",
    )

    solver_pade = HEOMSolver(H, [bathL, bathR], max_depth=2, options=options)
    rho_ss_pade, ado_ss_pade = solver_pade.steady_state()
    current = state_current(ado_ss_pade, bath_tag="R")

    progress.value += 1
    return np.real(current)


curr_ss_analytic_thetas = [
    current_analytic_for_theta(e1, bath_L, bath_R, theta)
    for theta in thetas
]

# The number of expansion terms has been dropped to Nk=6 to speed
# up notebook execution. Increase to Nk=10 for more accurate results.
curr_ss_pade_theta = [
    current_pade_for_theta(H, bath_L, bath_R, theta, Nk=6)
    for theta in thetas
]
```

Below we plot the results and see that even with `Nk=6`, the HEOM Pade approximation gives good results for the steady state current. Increasing `Nk` to `10` gives very accurate results.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(
    thetas, 2.434e-4 * 1e6 * np.array(curr_ss_analytic_thetas),
    color="black", linewidth=3,
    label=r"Analytical",
)
ax.plot(
    thetas, 2.434e-4 * 1e6 * np.array(curr_ss_pade_theta),
    'r--', linewidth=3,
    label=r"HEOM Pade $N_k=10$, $n_{\mathrm{max}}=2$",
)


ax.locator_params(axis='y', nbins=4)
ax.locator_params(axis='x', nbins=4)

ax.set_xticks([-2.5, 0, 2.5])
ax.set_xticklabels([-2.5, 0, 2.5])
ax.set_xlabel(r"Bias voltage $\Delta \mu$ ($V$)", fontsize=28)
ax.set_ylabel(r"Current ($\mu A$)", fontsize=28)
ax.legend(fontsize=25);
```

## About

```{code-cell} ipython3
qutip.about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell} ipython3
assert np.allclose(curr_ss_pade_L + curr_ss_pade_R, 0)
assert np.allclose(curr_ss_mats_L + curr_ss_mats_R, 0)
assert np.allclose(curr_ss_pade_R, curr_ss_analytic, rtol=1e-4)
```
