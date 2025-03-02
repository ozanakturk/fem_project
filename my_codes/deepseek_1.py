from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile

# Parameters
alpha = 1.0
beta = 1.0
T = 2.0  # Final time
num_steps = 50  # Number of time steps
dt = T / num_steps  # Time step size

# Create mesh and function space
msh = create_unit_square(MPI.COMM_WORLD, 10, 10)
V = functionspace(msh, ("Lagrange", 1))

# Define boundary condition
u_D = fem.Function(V)
u_D.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * 0.0)

# Define initial condition
u_n = fem.Function(V)
u_n.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * 0.0)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, beta - 2 - 2 * alpha)

F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
a = ufl.lhs(F)
L = ufl.rhs(F)

# Time-stepping
uh = fem.Function(V)
t = 0
for n in range(num_steps):
    t += dt
    u_D.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * t)
    
    # Update boundary condition
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))
    # Assemble and solve
    A = fem.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = fem.assemble_vector(L)
    fem.apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=MPI.SUM, mode=MPI.REPLACE)
    fem.set_bc(b, [bc])
    
    solver = fem.petsc.LinearSolver(msh.comm, A, "cg", "petsc_amg")
    solver.solve(uh.vector, b)
    
    # Update previous solution
    u_n.x.array[:] = uh.x.array

# Compare with the known solution
u_exact = fem.Function(V)
u_exact.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * T)

# Compute error
error = fem.assemble_scalar(fem.form((uh - u_exact)**2 * ufl.dx))
error = np.sqrt(msh.comm.allreduce(error, op=MPI.SUM))

print(f"L2 error: {error}")

# Save solution to file
with XDMFFile(msh.comm, "heat_equation.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

