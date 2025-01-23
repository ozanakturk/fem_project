"""
Literature:
[1] Farrell, Patrick E., Robert C. Kirby, and Jorge Marchena-Menendez. "Irksome: Automating Runge--Kutta time-stepping for finite element methods." arXiv preprint arXiv:2006.16282 (2020).
"""
from fenics import *

from irksome import GaussLegendre, RadauIIA, LobattoIIIC

from ufl.log import error

from ufl.algorithms.ad import expand_derivatives

import numpy as np


def heatreplace(v, u, k):
    # defines heat equation without RHS, change to any other equation
    L = (inner(k, v) * dx + inner(grad(u), grad(v)) * dx)
    return L


def mul(one, other):
    return MixedFunctionSpace((one, other))


T = 2.0            # final time
t = 0              # current time
num_steps = 4     # number of time steps
dt = T / num_steps  # time step size
iterations = 10
konvX = []
konvY = []
for ts in range(iterations):
    num_steps = num_steps*2
    dt = T / num_steps
    t = 0
    konvX.append(dt)
    alpha = 3          # parameter alpha
    beta = 1.2         # parameter beta

    # define geometry and spatial discretization
    nx = ny = 8
    msh = UnitSquareMesh(nx, ny)

    # get RK scheme
    bt = LobattoIIIC(3)
    num_stages = bt.num_stages
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
    du_Ddt = num_stages * [None]
    bc = []
    for i in range(num_stages):
        du_Ddt[i] = Expression('3*beta*t*t', degree=2,
                               alpha=alpha, beta=beta, t=0)
        du_Ddt[i].t = t
        for j in range(i-1):
            du_Ddt[i].t = du_Ddt[i].t + bt.c[j] * dt
        if(num_stages == 1):
            bc.append(DirichletBC(Vbig, du_Ddt[i], boundary))
        else:
            bc.append(DirichletBC(Vbig.sub(i), du_Ddt[i], boundary))

    # Define initial condition
    u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t*t*t',
                     degree=2, alpha=alpha, beta=beta, t=0)
    u_ini = interpolate(u_D, V)

    # Define problems rhs.
    f = num_stages * [None]
    for i in range(num_stages):
        f[i] = Expression('3*beta*t*t- 2 - 2*alpha',
                          degree=2, alpha=alpha, beta=beta, t=0)
        f[i].t = t + bt.c[i] * dt

    k = TrialFunction(Vbig)
    v = TestFunction(Vbig)

    ks = split(k)
    vs = split(v)

    # Define solutions per stage
    u = num_stages * [None]
    for i in range(num_stages):
        uhelp = u_ini
        for j in range(num_stages):
            uhelp = uhelp + A[i][j] * dt * ks[j]
        u[i] = uhelp

    rh = 0
    for i in range(num_stages):
        rh = rh + f[i] * vs[i]*dx

    # Assemble weak form
    F = 0
    for i in range(num_stages):
        F = F + heatreplace(vs[i], u[i], ks[i])
    F = F-rh
    a, L = lhs(F), rhs(F)

    # Unknown: stages k
    k = Function(Vbig)

    arrayY = []
    arrayX = []
    t = 0

    for n in range(num_steps):

        # Update BCs and rhs wrt current time.
        for i in range(num_stages):
            du_Ddt[i].t = t + bt.c[i] * dt
            f[i].t = t + bt.c[i] * dt

        # Compute solution for stages
        solve(a == L, k, bc)
        # Assemble solution from stages
        if(num_stages == 1):
            u_sol_help = u_ini + dt*bt.b[0]*k
        else:
            u_sol_help = u_ini
            for i in range(num_stages):
                u_sol_help = u_sol_help + dt*bt.b[i]*k.sub(i)
        u_sol = project(u_sol_help, V)

        u_ini.assign(u_sol)
        # Update time and compute reference solution
        t += dt
        u_D.t = t
        u_ref = interpolate(u_D, V)
        # Compute error
        error_normalized = (u_ref - u_sol) / u_ref
        error_pointwise = project(abs(error_normalized), V)
        # determine L2 norm to estimate total error
        error_total = sqrt(
            assemble(inner(error_pointwise, error_pointwise) * dx))
        error_pointwise.rename("error", " ")
        print('t = %.2f: error = %.3g' % (t, error_total))
        # Compute error at vertices
        arrayY.append(error_total)
        arrayX.append(t)
    konvY.append(max(arrayY))

# save to file
with open('konvergenz.txt', 'w') as file:
    for i in range(iterations):
        file.write('({},{})\n'.format(konvX[i], konvY[i]))
