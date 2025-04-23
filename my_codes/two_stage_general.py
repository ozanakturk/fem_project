from petsc4py import PETSc
from mpi4py import MPI
import ufl.split_functions
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import ufl
import numpy
t = 0  # Start time
T = 2  # End time
num_steps = 600  # Number of time steps
dt = (T - t) / num_steps  # Time step size
alpha = 3
beta = 1.2

# Boolean value for creating .xdmf file
write = 0

# Butcher Tableau

"""
c1 | a11 a12
c2 | a21 a22
-------------
   | b1  b2 

0 |  0    0
1 |  1    0
-------------
  | 1/2  1/2 
"""

bt_a11 = 0.5
bt_a12 = 0.0
bt_a21 = 0.0
bt_a22 = 0.5

bt_b1 = 0
bt_b2 = 1

bt_c1 = 0.5
bt_c2 = 0.5

nx, ny = 5, 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

# Mixed function space
V = fem.functionspace(domain, ("Lagrange", 1))
from basix.ufl import mixed_element
mixed = mixed_element([V.ufl_element()] * 2)
Vbig = fem.functionspace(domain, mixed)

class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
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

du_Ddt_help = boundary_condition(alpha, beta, t)

du_D_dt = fem.Function(Vbig)
du_D_dt.sub(0).interpolate(du_Ddt_help)
du_D_dt.sub(1).interpolate(du_Ddt_help)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(du_D_dt, fem.locate_dofs_topological(Vbig, fdim, boundary_facets))

# As $f$ is a constant independent of $t$, we can define it as a constant.

class source_term():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return self.beta - 2 - 2 * alpha + x[0] * 0

f_help = source_term(alpha, beta, t)

f = fem.Function(Vbig)
f.sub(0).interpolate(f_help)
f.sub(1).interpolate(f_help)
#f = fem.Constant(domain, beta - 2 - 2 * alpha)

xdmf = io.XDMFFile(domain.comm, "my_codes/heat_paraview/two-stage-general.xdmf", "w")
xdmf.write_mesh(domain)

u_n = fem.Function(V, name = "u_n") # Vbig
u_n.interpolate(u_exact)

### Save the function u_n, as this is the complete solution. uh is only k
xdmf.write_function(u_n, t)

k = ufl.TrialFunctions(Vbig)
v = ufl.TestFunction(Vbig)
kh = fem.Function(Vbig, name = "kh") 

P = k[0] * v[0] * ufl.dx + \
      dt * (bt_a11 * ufl.dot(ufl.grad(k[0]), ufl.grad(v[0])) + bt_a12 * ufl.dot(ufl.grad(k[0]), ufl.grad(v[0]))) * ufl.dx\
        - f.sub(0) * v[0] * ufl.dx + ufl.dot(ufl.grad(u_n), ufl.grad(v[0])) * ufl.dx \
          + k[1] * v[1] * ufl.dx + \
            dt * (bt_a21 * ufl.dot(ufl.grad(k[0]), ufl.grad(v[1])) + bt_a22 * ufl.dot(ufl.grad(k[1]), ufl.grad(v[1]))) * ufl.dx\
                - f.sub(1) * v[1] * ufl.dx + ufl.dot(ufl.grad(u_n), ufl.grad(v[1])) * ufl.dx \

a = fem.form(ufl.lhs(P))
L = fem.form(ufl.rhs(P))

A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = create_vector(L)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

for n in range(num_steps):
    u_exact.t += dt
    u_maxError.interpolate(u_exact)

    # Update Diriclet boundary condition
    du_Ddt_help.t += dt * bt_c1
    du_D_dt.sub(0).interpolate(du_Ddt_help)
    du_Ddt_help.t += - dt * bt_c1 + dt * bt_c2
    du_D_dt.sub(1).interpolate(du_Ddt_help)
    du_Ddt_help.t += - dt * bt_c2 + dt
    
    # Update source term
    f_help.t +=  dt * bt_c1
    f.sub(0).interpolate(f_help)
    f_help.t += - dt * bt_c1 + dt * bt_c2
    f.sub(1).interpolate(f_help)
    f_help.t += - dt * bt_c2 + dt
    
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    
    solver.solve(b, kh.x.petsc_vec)
    kh.x.scatter_forward()
    
    ###Updateing the previous step

    u_n.x.array[:] += bt_b1 * dt * kh.x.array[0] + bt_b2 * dt * kh.x.array[1]

    ### Write complete solution of every time step to xdmf file
    if write == 1:
        xdmf.write_function(u_n, u_exact.t)
    

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