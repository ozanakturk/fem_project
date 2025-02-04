import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io
from ufl import TrialFunction, TestFunction, grad, dot, dx, Function, FunctionSpace

# Define parameters
alpha = 1.0  # Thermal diffusivity
T = 2.0      # Final time
num_steps = 10  # Number of time steps
dt = T / num_steps  # Time step size

# Create 2D mesh and function space
nx, ny = 30, 30
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.FunctionSpace(domain, ("CG", 1))

# Define boundary condition
u_D = fem.Constant(domain, PETSc.ScalarType(0.0))

def boundary(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)) | np.logical_or(
        np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
    )

bc = fem.dirichletbc(value=u_D, facets=fem.locate_dofs_geometrical(V, boundary))

# Initial condition
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# Define trial and test functions
v = TestFunction(V)

# Define the heat equation's spatial term
def heat_residual(u):
    return -alpha * dot(grad(u), grad(v)) * dx

# Time-stepping
u = fem.Function(V)  # Solution at the current step
t = 0

for n in range(num_steps):
    # Update current time
    t += dt

    # RK4 intermediate stages
    k1 = fem.assemble_vector(fem.form(heat_residual(u_n)))
    u1 = fem.Function(V)
    u1.vector.axpy(dt / 2, k1)
    u1.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    k2 = fem.assemble_vector(fem.form(heat_residual(u1)))
    u2 = fem.Function(V)
    u2.vector.axpy(dt / 2, k2)
    u2.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    k3 = fem.assemble_vector(fem.form(heat_residual(u2)))
    u3 = fem.Function(V)
    u3.vector.axpy(dt, k3)
    u3.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    k4 = fem.assemble_vector(fem.form(heat_residual(u3)))
    u.vector[:] = u_n.vector + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Apply boundary conditions
    fem.petsc.apply_lifting(u.vector, [bc], [1.0])
    fem.petsc.set_bc(u.vector, bc)

    # Update solution
    u_n.x.array[:] = u.x.array

    # Save or log results
    with io.XDMFFile(MPI.COMM_WORLD, f"heat_solution_t{n}.xdmf", "w") as file:
        file.write_mesh(domain)
        file.write_function(u)

# End of simulation
