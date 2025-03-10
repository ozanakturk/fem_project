from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import create_matrix, assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import ufl
from petsc4py import PETSc

# Time stepping parameters
T = 2.0            # final time
t = 0.0            # current time
num_steps = 40     # number of time steps
dt = T / num_steps  # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Define geometry and spatial discretization
nx = ny = 8
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

# Get Runge-Kutta scheme (assuming LobattoIIIC(3) is defined elsewhere)
#bt = LobattoIIIC(3)  # Make sure this is defined
"""
# Heun's method two-stage
A = [[0, 0], [1, 0]] 
#A = [[0, 0], [1, 0]]
b = [1/2, 1/2]
c = [0, 1]
"""

A = [[0]] 
#A = [[0, 0], [1, 0]]
b = [1]
c = [0]
num_stages = len(c)

# Create mixed function space depending on the number of stages
V = fem.functionspace(msh, ("CG", 1))

if num_stages == 1:
    Vbig = V
else:
    import basix
    mixed_element = basix.ufl.mixed_element(num_stages * [V.ufl_element()])
    Vbig = fem.functionspace(msh, mixed_element)

# Define boundary condition function
def boundary(x):
    return np.full(x.shape[1], True)  # True for all boundary points
"""
# Define boundary conditions (Dirichlet BC)
du_Ddt = ns * []
bc = []
for i in range(ns):
    # Define a function to hold the boundary condition
    du_Ddt_i = fem.Function(V)
    t_new = t + c[i] * dt
    du_Ddt_i.interpolate(lambda x: np.full(x.shape[1], 2 * beta * t_new))
    du_Ddt.append(du_Ddt_i)

    # Apply boundary condition
    #bc.append(fem.dirichletbc(du_Ddt_i, fem.locate_dofs_geometrical(Vbig.sub(i), boundary)))
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc.append(fem.dirichletbc(du_Ddt_i, fem.locate_dofs_topological(Vbig.sub(i), fdim, boundary_facets)))
"""

# Define boundary conditions (Dirichlet BC)
class boundary_cond():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return self.beta
bound_1 = boundary_cond(alpha, beta, t)

du_Ddt = num_stages * [None]
bc = []
for i in range(num_stages):
    # Define a function to hold the boundary condition
    if num_stages == 1:
        du_Ddt[i] = fem.Function(V)
    else:
        du_Ddt[i] = fem.Function(Vbig.sub(i))
    du_Ddt[i].interpolate(lambda x: np.full(x.shape[1], beta))
    for j in range (i-1):
        du_Ddt[i].t = du_Ddt[i].t + c[j] * dt
    
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    if(num_stages==1):
        bc.append(fem.dirichletbc(du_Ddt[i], fem.locate_dofs_topological(Vbig, fdim, boundary_facets)))
    else:
        bc.append(fem.dirichletbc(du_Ddt[i], fem.locate_dofs_topological(Vbig.sub(i), fdim, boundary_facets)))

# Define initial condition
u_ini = fem.Function(V)
u_ini.interpolate(lambda x: 1 + x[0]**2 + alpha * x[1]**2 + beta * t**2)

# Define right-hand side function f
f = []
if num_stages == 1:
    f = fem.Constant(msh, beta - 2 - 2 * alpha)
else:
    for i in range(num_stages):
        f_i = fem.Function(V)
        t_new = t + c[i] * dt
        f_i.interpolate(lambda x: np.full(x.shape[1], beta - 2 - 2 * alpha))
        f.append(f_i)

#########################################

# Define weak form
if num_stages == 1:
    k = ufl.TrialFunction(Vbig)
    v = ufl.TestFunction(Vbig)

    # Define solutions per stage (generalized)
    u_stages = u_ini# + A[0][0] * dt * k     
    F = (ufl.dot(k, v) * ufl.dx) + (dt * ufl.dot(ufl.grad(u_stages), ufl.grad(v)) * ufl.dx) - (ufl.dot(u_stages, v) * ufl.dx) - (ufl.dot(f, v) * ufl.dx)    
else:
    k = ufl.TrialFunctions(Vbig)
    v = ufl.TestFunctions(Vbig)

    # Define solutions per stage (generalized)
    u_stages = [u_ini + sum(A[i][j] * dt * k[j] for j in range(num_stages)) for i in range(num_stages)]
    F = sum(ufl.dot(k[i], v[i]) * ufl.dx + ufl.dot(ufl.grad(u_stages[i]), ufl.grad(v[i])) * ufl.dx - f[i] * v[i] * ufl.dx
            for i in range(num_stages))

a, L = fem.form(ufl.lhs(F)), fem.form(ufl.rhs(F))

# Create PETSc matrix and vector
A_mat = create_matrix(a)
b_vec = create_vector(L)

# Set up linear solver
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A_mat)
solver.setType(PETSc.KSP.Type.GMRES)  # Choose GMRES or another solver type
solver.getPC().setType(PETSc.PC.Type.HYPRE)  # Use Hypre preconditioner

# Solution function for stages
k_sol = fem.Function(Vbig)

# Output file
with io.XDMFFile(MPI.COMM_WORLD, "heat_gaussian/solution.xdmf", "w") as vtkfile:

    # Time-stepping loop
    for n in range(num_steps):
        # Update BCs and RHS
        if num_stages == 1:
            t_new = t + c[0] * dt
            du_Ddt[0].interpolate(lambda x: np.full(x.shape[1], beta))
            # f.interpolate...
        else:
            for i in range(num_stages):
                t_new = t + c[i] * dt
                du_Ddt[i].interpolate(lambda x: np.full(x.shape[1], beta))
                f[i].interpolate(lambda x: np.full(x.shape[1], beta - 2 - 2 * alpha))

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
        u_sol.interpolate(lambda x: u_ini.x.array + dt * (b[0] * k_sol.x.array))

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

        print(f"t = {t:.2f}: error = {error_L2:.3g}")

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