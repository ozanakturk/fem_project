# -*- coding: utf-8 -*-
"""
Literature:
[1] Farrell, Patrick E., Robert C. Kirby, and Jorge Marchena-Menendez. "Irksome: Automating Runge--Kutta time-stepping for finite element methods." arXiv preprint arXiv:2006.16282 (2020).
"""

from fenics import *

from irksome import GaussLegendre, RadauIIA, LobattoIIIC

from ufl.log import error

from ufl.algorithms.ad import expand_derivatives

import numpy as np

T = 2.0            # final time
t = 0              # current time
num_steps = 40     # number of time steps
dt = T / num_steps  # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# define geometry and spatial discretization
nx = ny = 8
msh = UnitSquareMesh(nx, ny)


# get RK scheme
bt = LobattoIIIC(3)
num_stages = bt.num_stages
ns = bt.num_stages
A = bt.A


# Create mixed function space depending on number of stages
V = FunctionSpace(msh, "P", 1)
if(num_stages == 1):
    Vbig = V
else:
    mixed = MixedElement(num_stages*[V.ufl_element()])
    Vbig = FunctionSpace(V.mesh(), mixed)

# Define boundary conditions


def boundary(x, on_boundary):
    return on_boundary


du_Ddt = ns * [None]
bc = []
for i in range(ns):
    du_Ddt[i] = Expression('2*beta*t', degree=2, alpha=alpha, beta=beta, t=0)
    du_Ddt[i].t = t + bt.c[i] * dt
    bc.append(DirichletBC(Vbig.sub(i), du_Ddt[i], boundary))

# Define initial condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t*t',
                 degree=2, alpha=alpha, beta=beta, t=0)
u_ini = interpolate(u_D, V)

# Define problems rhs. Important: If f is time dependent, we need the same procedure like for the boundary conditions.
f = ns * [None]
for i in range(ns):
    f[i] = Expression('2*beta*t - 2 - 2*alpha', degree=2,
                      alpha=alpha, beta=beta, t=0)
    f[i].t = t + bt.c[i] * dt

k0, k1 = TrialFunctions(Vbig)
v0, v1 = TestFunctions(Vbig)

# Define solutions per stage. Todo: Should be generalized via a for-loop
u0 = u_ini + A[0][0] * dt * k0 + A[0][1] * dt * k1
u1 = u_ini + A[1][0] * dt * k0 + A[1][1] * dt * k1

# Assemble weak form. Todo: Should be generalized via for-loop
F = (inner(k0, v0) * dx + inner(grad(u0), grad(v0)) * dx) + (inner(k1, v1)
                                                             * dx + inner(grad(u1), grad(v1)) * dx) - f[1] * v1 * dx - f[0] * v0 * dx
a, L = lhs(F), rhs(F)

vtkfile = File("heat_gaussian/solution.pvd")

# Unknown: stages k
k = Function(Vbig)


arrayY = []
arrayX = []

for n in range(num_steps):

    # Update BCs and rhs wrt current time.
    for i in range(ns):
        du_Ddt[i].t = t + bt.c[i] * dt
        f[i].t = t + bt.c[i] * dt

    # Compute solution for stages
    solve(a == L, k, bc)
    # Assemble solution from stages
    u_sol = project(u_ini + dt * (bt.b[0] * k.sub(0) + bt.b[1] * k.sub(1)), V)
    # Update initial condition with solution
    u_ini.assign(u_sol)
    # Update time and compute reference solution
    t += dt
    u_D.t = t
    u_ref = interpolate(u_D, V)
    # Compute error
    error_normalized = (u_ref - u_sol) / u_ref
    error_pointwise = project(abs(error_normalized), V)
    # determine L2 norm to estimate total error
    error_total = sqrt(assemble(inner(error_pointwise, error_pointwise) * dx))
    error_pointwise.rename("error", " ")
    print('t = %.2f: error = %.3g' % (t, error_total))
