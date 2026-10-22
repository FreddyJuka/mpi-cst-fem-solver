#!/usr/bin/env python3
"""Derive and verify the Kirsch displacement field symbolically (sympy).

Starting from the SPEC section 8 stresses (uniaxial sigma_inf along x, hole radius a,
theta from the x axis), the strains follow from Hooke's law written with the shear modulus
mu and the Kolosov constant kappa (plane stress: kappa = (3 - nu)/(1 + nu); plane strain:
kappa = 3 - 4 nu):

    eps_rr  = [(kappa + 1) s_rr - (3 - kappa) s_tt] / (8 mu)
    eps_tt  = [(kappa + 1) s_tt - (3 - kappa) s_rr] / (8 mu)
    gam_rt  = s_rt / mu

and are integrated: u_r = int eps_rr dr + f(theta); u_t from eps_tt = (u_r + d u_t/d theta)/r;
the shear strain fixes f and g(r) up to a rigid-body motion, which the symmetry conditions
(u_x = 0 on x = 0, u_y = 0 on y = 0) remove.

Checks (exit status 1 if any fails), for both plane conditions with E and nu symbolic:
  (a) strains of the derived displacements, through Hooke's law, give the SPEC stresses;
  (b) those stresses satisfy equilibrium (no body force);
  (c) u_x = 0 on x = 0 and u_y = 0 on y = 0;
  plus: traction-free hole (s_rr = s_rt = 0 at r = a).
Prints the expressions and reference values (Cartesian u_x, u_y) used by the C++ unit test.
"""
import sys

import sympy as sp

r, th, a, s, E, nu = sp.symbols("r theta a sigma_inf E nu", positive=True)
mu, kappa = sp.symbols("mu kappa", positive=True)

# ---- SPEC section 8 stresses
q2, q4 = a**2 / r**2, a**4 / r**4
s_rr = s / 2 * (1 - q2) + s / 2 * (1 - 4 * q2 + 3 * q4) * sp.cos(2 * th)
s_tt = s / 2 * (1 + q2) - s / 2 * (1 + 3 * q4) * sp.cos(2 * th)
s_rt = -s / 2 * (1 + 2 * q2 - 3 * q4) * sp.sin(2 * th)

failures = []


def check(name, expr):
    ok = sp.simplify(expr) == 0
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        failures.append(name)
        print("        residual:", sp.simplify(expr))


# ---- Integration with (mu, kappa)
e_rr = ((kappa + 1) * s_rr - (3 - kappa) * s_tt) / (8 * mu)
e_tt = ((kappa + 1) * s_tt - (3 - kappa) * s_rr) / (8 * mu)
g_rt = s_rt / mu

ur0 = sp.integrate(sp.expand(e_rr), r)
ut0 = sp.integrate(sp.expand(r * e_tt - ur0), th)
# Remaining compatibility: gam_rt = (1/r) du_r/dth + du_t/dr - u_t/r must hold up to a
# rigid-body motion; the residual of the particular solution must vanish.
residual = sp.simplify(g_rt - (sp.diff(ur0, th) / r + sp.diff(ut0, r) - ut0 / r))
print("shear compatibility residual of the particular solution:", residual)
if residual != 0:
    failures.append("particular solution does not satisfy the shear strain")

# General solution = particular + rigid-body motion (translations tx, ty; rotation w).
tx, ty, w = sp.symbols("t_x t_y omega")
u_r = ur0 + tx * sp.cos(th) + ty * sp.sin(th)
u_t = ut0 - tx * sp.sin(th) + ty * sp.cos(th) + w * r
u_x = u_r * sp.cos(th) - u_t * sp.sin(th)
u_y = u_r * sp.sin(th) + u_t * sp.cos(th)

# Symmetry: u_x(theta = pi/2) = 0 and u_y(theta = 0) = 0 for all r.
eqs = []
for expr in (u_x.subs(th, sp.pi / 2), u_y.subs(th, 0)):
    num = sp.numer(sp.together(sp.expand(expr)))
    eqs += sp.Poly(sp.expand(num), r).coeffs()
sol = sp.solve(eqs, [tx, ty, w], dict=True)
if len(sol) != 1:
    failures.append(f"rigid-body constants not uniquely determined: {sol}")
    sol = [{tx: 0, ty: 0, w: 0}]
print("rigid-body constants from symmetry:", sol[0])
u_r = sp.simplify(u_r.subs(sol[0]))
u_t = sp.simplify(u_t.subs(sol[0]))

print("\nDisplacement field (mu = E / (2 (1 + nu))):")
print("  u_r     =", sp.factor_terms(sp.collect(sp.expand(u_r), [sp.cos(2 * th)])))
print("  u_theta =", sp.factor_terms(sp.expand(u_t)))

# ---- Verification for each plane condition with E, nu symbolic
conditions = {
    "plane_stress": {mu: E / (2 * (1 + nu)), kappa: (3 - nu) / (1 + nu)},
    "plane_strain": {mu: E / (2 * (1 + nu)), kappa: 3 - 4 * nu},
}
for name, sub in conditions.items():
    print(f"\nChecks, {name} (kappa = {sub[kappa]}):")
    ur, ut = u_r.subs(sub), u_t.subs(sub)
    # Strains from displacements (small strain, polar).
    err = sp.diff(ur, r)
    ett = (ur + sp.diff(ut, th)) / r
    grt = sp.diff(ur, th) / r + sp.diff(ut, r) - ut / r
    # Hooke's law in E, nu (independent of the kappa form used for the integration).
    if name == "plane_stress":
        c = E / (1 - nu**2)
        srr = c * (err + nu * ett)
        stt = c * (ett + nu * err)
    else:
        c = E / ((1 + nu) * (1 - 2 * nu))
        srr = c * ((1 - nu) * err + nu * ett)
        stt = c * ((1 - nu) * ett + nu * err)
    srt = E / (2 * (1 + nu)) * grt
    check("(a) sigma_rr from displacements = SPEC sigma_rr", srr - s_rr)
    check("(a) sigma_tt from displacements = SPEC sigma_tt", stt - s_tt)
    check("(a) sigma_rt from displacements = SPEC sigma_rt", srt - s_rt)
    check("(b) radial equilibrium", sp.diff(srr, r) + sp.diff(srt, th) / r + (srr - stt) / r)
    check("(b) tangential equilibrium", sp.diff(stt, th) / r + sp.diff(srt, r) + 2 * srt / r)
    ux = ur * sp.cos(th) - ut * sp.sin(th)
    uy = ur * sp.sin(th) + ut * sp.cos(th)
    check("(c) u_x = 0 on x = 0 (theta = pi/2)", ux.subs(th, sp.pi / 2))
    check("(c) u_y = 0 on y = 0 (theta = 0)", uy.subs(th, 0))
    check("traction-free hole: sigma_rr(r = a) = 0", srr.subs(r, a))
    check("traction-free hole: sigma_rt(r = a) = 0", srt.subs(r, a))

# ---- Reference values for the C++ unit test
vals = {s: sp.Integer(100) * 10**6, a: sp.Rational(1, 10), E: sp.Integer(210) * 10**9,
        nu: sp.Rational(3, 10)}
points = [(sp.Rational(1, 10), 0), (0, sp.Rational(1, 10)), (sp.Rational(3, 10), sp.Rational(1, 5)),
          (sp.Rational(1, 1), sp.Rational(1, 1)), (sp.Rational(1, 1), 0), (sp.Rational(1, 20),
                                                                           sp.Rational(9, 10))]
print("\nReference values (sigma_inf = 100e6, a = 0.1, E = 210e9, nu = 0.3):")
print("  plane, x, y, u_x, u_y")
for name, sub in conditions.items():
    ux = (u_r * sp.cos(th) - u_t * sp.sin(th)).subs(sub)
    uy = (u_r * sp.sin(th) + u_t * sp.cos(th)).subs(sub)
    for x, y in points:
        at = {r: sp.sqrt(x**2 + y**2), th: sp.atan2(y, x), **vals}
        print(f"  {name}, {float(x)}, {float(y)}, {sp.N(ux.subs(at), 20)}, "
              f"{sp.N(uy.subs(at), 20)}")

if failures:
    print("\nFAILED:", *failures, sep="\n  - ")
    sys.exit(1)
print("\nAll symbolic checks passed.")
