# https://www.firedrakeproject.org/demos/DG_advection.py.html

from petsc4py import PETSc
from mpi4py import MPI
import ufl.finiteelement
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import ufl
import numpy as np
import time

t = 0  # Start time
T = 2  # End time
num_steps = 600  # Number of time steps (Boundary: 561)
dt = (T - t) / num_steps  # Time step size
alpha = 3
beta = 1.2

# Butcher Table

A = [[1/2, -1/2], [1/2, 1/2]]
#A = [[0, 0], [1, 0]]
b = [1/2, 1/2]
c = [0, 1]
num_stages = 2


"""
A = np.array([[5/12, -1/12], [3/4, 1/4]])
b = np.array([3/4, 1/4])
c = np.array([1/3, 1])
num_stages = len(b)
"""

"""
A = np.array([[0,0], [0,0]])
b = np.array([0, 0])
c = np.array([0, 0])
num_stages = len(b)
"""
# As for the previous problem, we define the mesh and appropriate function spaces.

nx, ny = 5, 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
#V = ufl.MixedFunctionSpace(domain, "Lagrange", 1)

# Mixed function space
from basix.ufl import mixed_element
mixed = mixed_element([V.ufl_element()] * num_stages)
Vbig = fem.functionspace(domain, mixed)

# ## Defining the exact solution
class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t


u_exact = exact_solution(alpha, beta, t)

# ## Defining the boundary condition

"""
du_Ddt = num_stages * [None]
bc = []
import basix
for i in range(num_stages):
    du_Ddt[i] = Expression('2*beta*t', degree=2, alpha=alpha, beta=beta, t=0)
    du_Ddt[i].t = t + c[i] * dt
    bc.append(DirichletBC(Vbig.sub(i), du_Ddt[i], boundary))
"""
u_D = fem.Function(Vbig)
u_D.sub(0).interpolate(u_exact)
u_D.sub(1).interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# ## Defining the variational formulation

u_n = fem.Function(V)
"""u_D = fem.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t*t',
                 degree=2, alpha=alpha, beta=beta, t=0)"""
u_n.interpolate(u_exact)

# As f is a constant independent of t, we can define it as a constant.

f0 = fem.Constant(domain, beta - 2 - 2 * alpha)
f1 = fem.Constant(domain, beta - 2 - 2 * alpha)

# We can now create our variational formulation, with the bilinear form `a` and  linear form `L`.###########################################################################
############################################################################################################################################################################

uh = fem.Function(V)

k0 = ufl.TrialFunction(V)
k1 = ufl.TrialFunction(V)
v0 = ufl.TestFunction(V)
v1 = ufl.TestFunction(V)

# Define solutions per stage. Todo: Should be generalized via a for-loop
u0 = u_n + A[0][0] * dt * k0 + A[0][1] * dt * k1
u1 = u_n + A[1][0] * dt * k0 + A[1][1] * dt * k1

# We can now create our variational formulation, with the bilinear form `a` and  linear form `L`.
F = k0 * v0 * ufl.dx + dt * ufl.dot(ufl.grad(u0), ufl.grad(v0)) * ufl.dx + k1 * v1 * ufl.dx + dt * ufl.dot(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - (u0 + dt * f0) * v0 * ufl.dx - (u0 + dt * f0) * v0 * ufl.dx

a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

k = ufl.TrialFunction(V)


for n in range(num_steps):
    # Create the matrix and vector for the linear problem
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = create_vector(L)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Update Diriclet boundary condition
    u_exact.t += dt
    u_D.sub(0).interpolate(u_exact)
    u_D.sub(1).interpolate(u_exact)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

"""
# Compute values at mesh vertices
error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")
"""

