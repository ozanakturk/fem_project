# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # A known analytical solution
# Author: Jørgen S. Dokken
#
# Just as for the [Poisson problem](./../chapter1/fundamentals_code), we construct a test problem which makes it easy to determine if the calculations are correct.
#
# Since we know that our first-order time-stepping scheme is exact for linear functions, we create a problem which has linear variation in time. We combine this with a quadratic variation in space. Therefore, we choose the analytical solution to be
# \begin{align}
# u = 1 + x^2+\alpha y^2 + \beta t
# \end{align}
# which yields a function whose computed values at the degrees of freedom will be exact, regardless of the mesh size and $\Delta t$ as long as the mesh is uniformly partitioned.
# By inserting this into our original PDE, we find that the right hand side $f=\beta-2-2\alpha$. The boundary value $u_d(x,y,t)=1+x^2+\alpha y^2 + \beta t$ and the initial value $u_0(x,y)=1+x^2+\alpha y^2$.
#
# We start by defining the temporal discretization parameters, along with the parameters for $\alpha$ and $\beta$.

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import ufl
import numpy
t = 0  # Start time
T = 2  # End time
num_steps = 20  # Number of time steps
dt = (T - t) / num_steps  # Time step size
alpha = 3
beta = 1.2

# Butcher Tableau

bt_a = 1.0
bt_b = 1.0
bt_c = 1.0

# As for the previous problem, we define the mesh and appropriate function spaces.

# +

nx, ny = 5, 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1)) #CG

# -

# ## Defining the exact solution
# As in the membrane problem, we create a Python-class to resemble the exact solution

class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        #return 1 + x[0]**2 + self.alpha * x[1]**2 + (2 + 2 * self.beta) * self.t
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t

u_exact = exact_solution(alpha, beta, t)

### Create function to compute maximum nodal error at the end
u_maxError = fem.Function(V)

class boundary_condition():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return self.beta + x[0] * 0
        #return 2 * self.beta * self.t + x[0] * 0
        #return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t

du_Ddt_help = boundary_condition(alpha, beta, t)

du_D_dt = fem.Function(V)
du_D_dt.interpolate(du_Ddt_help)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(du_D_dt, fem.locate_dofs_topological(V, fdim, boundary_facets))

# As $f$ is a constant independent of $t$, we can define it as a constant.

class source_term():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        #return x[0] * 0
        return self.beta - 2 - 2 * alpha + x[0] * 0
        return 2 * self.beta * self.t - 2 - 2 * self.alpha + x[0] * 0

f_help = source_term(alpha, beta, t)

f = fem.Function(V)
f.interpolate(f_help)
# f = fem.Constant(domain, beta - 2 - 2 * alpha)

xdmf = io.XDMFFile(domain.comm, "heat_paraview/heat_edit.xdmf", "w")
xdmf.write_mesh(domain)

u_n = fem.Function(V, name = "u_n")
u_n.interpolate(u_exact)

k = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
uh = fem.Function(V, name = "uh")

### Save the function u_n, as this is the complete solution. uh is only k
xdmf.write_function(u_n, t)

def problem(u):
    return u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx + ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

a = fem.form(ufl.lhs(problem(k)))
L = fem.form(ufl.rhs(problem(k)))

A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = create_vector(L)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

for n in range(num_steps):
    # Update Diriclet boundary condition
    du_Ddt_help.t += dt * bt_c
    du_D_dt.interpolate(du_Ddt_help)

    ### Interpolate exact solution for each time step into function
    ### for maximun nodal error
    u_exact.t += dt
    u_maxError.interpolate(u_exact)

    # Update source term
    f_help.t +=  dt * bt_c
    f.interpolate(f_help)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    u_n.x.array[:] += dt * bt_b * uh.x.array

    ### Write complete solution of every time step to xdmf file
    xdmf.write_function(u_n, du_Ddt_help.t)

xdmf.close()


V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
### Computer the L2 error of the solution, therefore, take u_n and not uh, as uh is only k1
error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((u_n - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
### Computer the maximum nodal error of the solution, therefore, take u_n and not uh, as uh is only k1
error_max = domain.comm.allreduce(numpy.max(numpy.abs(u_n.x.array - u_maxError.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")