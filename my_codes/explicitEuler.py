from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import create_matrix, assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import ufl
from petsc4py import PETSc

# Time stepping parameters
T = 2.0            # final time
t = 0.0            # current time
num_steps = 500     # number of time steps
dt = T / num_steps  # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Define geometry and spatial discretization
nx = ny = 8
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

# Get Runge-Kutta scheme (assuming LobattoIIIC(3) is defined elsewhere)
num_stages = 1

# Create mixed function space depending on the number of stages
V = fem.functionspace(msh, ("CG", 1))
"""
du_Ddt = fem.Function(V)
du_Ddt.interpolate(lambda x: np.full(x.shape[1], 2 * beta * t))

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
bc_dt = fem.dirichletbc(du_Ddt, fem.locate_dofs_topological(V, fdim, boundary_facets))
"""
u_D = fem.Function(V)
u_D.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * t)

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
bc = []
bc.append(fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets)))

# Define initial condition
u_ini = fem.Function(V)
u_ini.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * t)

# Define right-hand side function f
f = fem.Constant(msh, beta - 2 - 2 * alpha)

#########################################

# Define weak form
k = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define solutions per stage (generalized)
u_stages = u_ini + 0 * dt * k   
F = (ufl.dot(k, v) * ufl.dx) + (dt * ufl.dot(ufl.grad(k), ufl.grad(v)) * ufl.dx) - (ufl.dot(u_stages, v) * ufl.dx) - (ufl.dot(f, v) * ufl.dx)
a, L = fem.form(ufl.lhs(F)), fem.form(ufl.rhs(F))
#a = fem.form(ufl.dot(k, v) * ufl.dx)
#L = fem.form(-dt * ufl.dot(ufl.grad(u_stages), ufl.grad(v)) * ufl.dx + ufl.dot(u_stages, v) * ufl.dx + ufl.dot(f, v) * ufl.dx)

# Create PETSc matrix and vector
A_mat = create_matrix(a)
b_vec = create_vector(L)

# Set up linear solver
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A_mat)
solver.setType(PETSc.KSP.Type.GMRES)  # Choose GMRES or another solver type
solver.getPC().setType(PETSc.PC.Type.HYPRE)  # Use Hypre preconditioner

"""# Output file
with io.XDMFFile(MPI.COMM_WORLD, "heat_gaussian/solution.xdmf", "w") as vtkfile:
"""
k_sol = fem.Function(V)

# Time-stepping loop
for n in range(num_steps):
    # Update BCs and RHS
    t += dt
    u_D.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * t)

    # Assemble system
    A_mat.zeroEntries()
    assemble_matrix(A_mat, a, bcs=bc)
    A_mat.assemble()
    assemble_vector(b_vec, L)
    apply_lifting(b_vec, [a], [bc])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_vec, bc)

    # Solve system
    solver.solve(b_vec, k_sol.x.petsc_vec)

    #print(k_sol.sub(0).x.array)
    
    # Assemble solution from stages
    u_sol = fem.Function(V)
    #u_sol.interpolate(lambda x: u_ini.x.array + dt * (b[0] * k_sol.sub(0).collapse().x.array + b[1] * k_sol.sub(1).collapse().x.array))
    u_sol.interpolate(lambda x: u_ini.x.array + dt * k_sol.x.array)

    # Write to file
    #vtkfile.write_function(u_sol, t)

    # Update initial condition
    u_ini.x.array[:] = u_sol.x.array[:]

    # Compute error
    t += dt
    u_ref = fem.Function(V)
    u_ref.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * t)

    error = fem.Function(V)
    error.interpolate(lambda x: np.abs((u_ref.x.array - u_sol.x.array) / u_ref.x.array))

    error_L2 = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(error, error) * ufl.dx)))

    #print(f"t = {t:.2f}: error = {error_L2:.3g}")

V_ex = fem.functionspace(msh, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * t)
error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form((k_sol - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if msh.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")


# Compute final L2 error
error_L2 = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_ref - u_sol, u_ref - u_sol) * ufl.dx)))

# Compute final maximum error (pointwise)
error_max = np.max(np.abs(u_ref.x.array - u_sol.x.array))

# Print results
print("\nFinal Errors at t = {:.2f}:".format(t))
print("-------------------------------------------------")
print("Overall L² Error   : {:.6g}".format(error_L2))
print("Maximum Error      : {:.6g}".format(error_max))
print("-------------------------------------------------")