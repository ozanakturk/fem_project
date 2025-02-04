from dolfinx import mesh, fem
from ufl import TrialFunction, TestFunction, inner, grad, dx, lhs, rhs
from ufl.finiteelement import FiniteElement
from basix.ufl import mixed_element
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem

# Problem parameters
T = 2.0            # final time
t = 0              # current time
num_steps = 40     # number of time steps
dt = T / num_steps  # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and function space
nx = ny = 8
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(msh, ("CG", 1))

# Define Lobatto IIIC Butcher Tableau manually
A = np.array([[5/12, -1/12], [3/4, 1/4]])
b = np.array([3/4, 1/4])
c = np.array([1/3, 1])
num_stages = len(b)

# Mixed function space
if num_stages == 1:
    Vbig = V
else:
    mixed = mixed_element([V.ufl_element()] * num_stages)
    Vbig = fem.functionspace(msh, mixed)

# Define boundary condition
def boundary(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1), np.isclose(x[1], 0), np.isclose(x[1], 1))

# Initialize boundary conditions and RHS terms
du_Ddt = [fem.Function(V) for _ in range(num_stages)]
bc = []
for i in range(num_stages):
    du_Ddt[i].interpolate(lambda x: 2 * beta * t)
    bc.append(fem.dirichletbc(du_Ddt[i], fem.locate_dofs_geometrical(Vbig.sub(i), boundary)))

# Initial condition
nu_ini = fem.Function(V)
nu_ini.interpolate(lambda x: 1 + x[0]**2 + alpha*x[1]**2 + beta*t**2)

# Define right-hand side
f = [fem.Function(V) for _ in range(num_stages)]
for i in range(num_stages):
    f[i].interpolate(lambda x: 2*beta*t - 2 - 2*alpha)

# Trial and test functions
k = TrialFunction(Vbig)
v = TestFunction(Vbig)

# Define stage solutions
nu_stages = [nu_ini + sum(A[i][j] * dt * k[j] for j in range(num_stages)) for i in range(num_stages)]

# Weak form
F = sum(inner(k[i], v[i]) * dx + inner(grad(nu_stages[i]), grad(v[i])) * dx - f[i] * v[i] * dx for i in range(num_stages))
a, L = lhs(F), rhs(F)

# Create solver
problem = LinearProblem(a, L, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# Solution function for Runge-Kutta stages
k_sol = fem.Function(Vbig)
u_sol = fem.Function(V)
u_ref = fem.Function(V)

# Time-stepping loop
for n in range(num_steps):
    # Update boundary conditions and RHS
    for i in range(num_stages):
        du_Ddt[i].interpolate(lambda x: 2 * beta * (t + c[i] * dt))
        f[i].interpolate(lambda x: 2 * beta * (t + c[i] * dt) - 2 - 2 * alpha)
    
    # Solve for Runge-Kutta stages
    problem.solve(k_sol)
    
    # Compute final solution from stages
    u_sol.interpolate(lambda x: nu_ini(x) + dt * sum(b[i] * k_sol.sub(i)(x) for i in range(num_stages)))
    
    # Update initial condition
    nu_ini.x.array[:] = u_sol.x.array
    
    # Update time
    t += dt
    u_ref.interpolate(lambda x: 1 + x[0]**2 + alpha*x[1]**2 + beta*t**2)
    
    # Compute error
    error_array = np.abs((u_ref.x.array - u_sol.x.array) / u_ref.x.array)
    error_total = np.sqrt(msh.comm.allreduce(np.sum(error_array**2), op=MPI.SUM))
    print(f't = {t:.2f}: error = {error_total:.3g}')
