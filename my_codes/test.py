from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, create_vector
import ufl
import numpy
import time

from rk_solver import rk_solver

t = 0  # Start time
T = 2  # End time
num_steps = 530  # Number of time steps # Default: 20
dt = (T - t) / num_steps  # Time step size
alpha = 3
beta = 1.2

# As for the previous problem, we define the mesh and appropriate function spaces.

nx, ny = 5, 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# ## Defining the exact solution
# As in the membrane problem, we create a Python-class to resemble the exact solution

class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t


u_exact = exact_solution(alpha, beta, t)

# ## Defining the boundary condition

u_D = fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# ## Defining the variational formulation

u_n = fem.Function(V)
u_n.interpolate(u_exact)

# As f is a constant independent of t, we can define it as a constant.

f = fem.Constant(domain, beta - 2 - 2 * alpha)

# We can now create our variational formulation, with the bilinear form `a` and  linear form `L`.

print()
a = 2
if a == 1:
    print("implicit")
    rk_solver("Backward Euler", domain, u_n, u_exact, u_D, bc, dt, num_steps, f, V)
elif a == 2:
    print("Heun")
    rk_solver("Heun", domain, u_n, u_exact, u_D, bc, dt, num_steps, f, V)
elif a == 3:
    print("Radau IIa")
    rk_solver("Radau IIa", domain, u_n, u_exact, u_D, bc, dt, num_steps, f, V)
else:
    print("RK4")
    rk_solver("RK4", domain, u_n, u_exact, u_D, bc, dt, num_steps, f, V)



# Explicit RK methods

"""F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx # Explicit
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = create_vector(L)
uh = fem.Function(V)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)"""

"""for method in ['Explicit Euler', 'Heun', 'RK4']:
    start_time = time.time()
    rk_solver(method, domain, u_n, u_exact, u_D, bc, a, b, solver, dt, num_steps, f, V, v)
    total_time = time.time() - start_time
  
    V_ex = fem.functionspace(domain, ("Lagrange", 2))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact)
    error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
    error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
    if domain.comm.rank == 0:
        print(f"Method name: " + method + "\t" 
              + f"Time step dt: {dt:.2e}" + "\t" 
              + f"L2-error: {error_L2:.2e}" + "\t" 
              + f"Error_max: {error_max:.2e}" + "\t"
              + f"Computation Time: {total_time:.2}" + " seconds" + "\n")"""

# Implicit RK methods

"""F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx # Implicit
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = create_vector(L)
uh = fem.Function(V)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)"""

"""for method in ['Backward Euler', 'Radau IIa', 'Lobatto IIIc', 'Gauss-Legendre']:
    start_time = time.time()
    rk_solver(method, domain, u_n, u_exact, u_D, bc, a, b, solver, dt, num_steps, f, V, v)
    total_time = time.time() - start_time

    V_ex = fem.functionspace(domain, ("Lagrange", 2))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact)
    error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
    error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
    if domain.comm.rank == 0:
        print(f"Method name: " + method + "\t" 
              + f"Time step dt: {dt:.2e}" + "\t" 
              + f"L2-error: {error_L2:.2e}" + "\t" 
              + f"Error_max: {error_max:.2e}" + "\t"
              + f"Computation Time: {total_time:.2}" + " seconds" + "\n")"""




