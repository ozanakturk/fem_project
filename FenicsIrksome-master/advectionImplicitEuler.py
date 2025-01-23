# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:31:10 2021

@author: nikol
"""

from fenics import *
from irksome import GaussLegendre, RadauIIA
from ufl.algorithms.ad import expand_derivatives
import numpy as np

# initialize Butcher Tableau
bt = RadauIIA(1)
A = bt.A
b = bt.b


# Define time, timesteps and parameters
T = 2.0                                             # final time
num_steps = 50                                      # number of time steps
dt = T / num_steps                                  # time step size

# define geometry and spatial discretization
nx = 10
msh = IntervalMesh(nx, 0, 1)

# define functionspace
V = FunctionSpace(msh, "CG", 1)

# Constant for advection equation
c = Constant(2.0)
# analytical solution for boundaries
u_D = Expression('x[0]-c*t', degree=1, c=c, t=0)

# define boundary condtitons


def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, u_D, boundary)  # boundary conditions

# assmeble weak form
u_n = interpolate(u_D, V)  # u at time 0
u = TrialFunction(V)  # Trialfunction
k0 = Function(V)
v0 = TestFunction(V)  # Testfunction
u0 = u_n + dt*Constant(1.0)*k0
F = u*v0*dx+u0*v0*dx - dt*c*Dx(u0, 0)*v0*k0*dx

# Residual
r = u - u_n + dt * c * Dx((u+u_n)/2, 0)
# Add SUPG stabilisation terms (from https://fenicsproject.org/qa/13458/how-implement-supg-properly-advection-dominated-equation/)
h = 0.1
tau = h/(2.0*c)  # tau from SUPG fenics example
F += tau * c * Dx(v0, 0) * r * dx
a, L = lhs(F), rhs(F)

# unknown function u
u = Function(V)
t = 0
arrayY = []
arrayX = []
for n in range(num_steps):
    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    u_ref = interpolate(u_D, V)
    error_normalized = (u_ref - u) / u_ref

    # project onto function space
    error_pointwise = project(abs(error_normalized), V)

    # determine L2 norm to estimate total error
    error_total = sqrt(assemble(inner(error_pointwise, error_pointwise) * dx))
    error_pointwise.rename("error", " ")
    print('t = %.2f: error = %.3g' % (t, error_total))
    arrayY.append(u.vector().get_local().max())
    arrayX.append(t)

    # Update previous solution
    u_n.assign(u)

# save to file
with open('advectionImplicitEuler.txt', 'w') as file:
    for i in range(num_steps):
        file.write('{},{}\n'.format(arrayX[i], arrayY[i]))
