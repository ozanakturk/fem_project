from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import ufl
import numpy


class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t
    

class boundary_condition():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return self.beta + x[0] * 0
    
class source_term():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return self.beta - 2 - 2 * self.alpha + x[0] * 0

class RK():
    def __init__(self, a, b, c): # Name, order
        # Time and constants
        self.t = 0  # Start time
        self.T = 2  # End time
        self.num_steps = 20  # Number of time steps
        self.dt = (self.T - self.t) / self.num_steps  # Time step size

        self.bt_a = a
        self.bt_b = b
        self.bt_c = c
        self.num_stages = len(c)

        self.domain = None
        self.V = None
        self.bc = None

        # Boundary and source term
        self.du_D_dt = None
        self.du_Ddt_help = None
        self.f = None
        self.f_help = None
        self.u_maxError = None

        # Exact solution and the initial condition (previous solution)
        self.u_exact = None
        self.u_n = None

        #
        self.xdmf = None

        self.k = None
        self.v = None
        self.uh = None

    def mesh(self, nx=5, ny=5, method="Lagrange", order=1):
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
        self.V = fem.functionspace(self.domain, (method, order)) #CG

        # Visualisation
        self.xdmf = io.XDMFFile(self.domain.comm, "heat_edit.xdmf", "w")
        self.xdmf.write_mesh(self.domain)

        ### Create function to compute maximum nodal error at the end
        self.u_maxError = fem.Function(self.V)

    def boundary(self, alpha=3, beta=1.2):
        # The boundary condition
        self.du_Ddt_help = boundary_condition(alpha, beta, self.t)
        self.du_D_dt = fem.Function(self.V)
        self.du_D_dt.interpolate(self.du_Ddt_help)
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        self.bc = fem.dirichletbc(self.du_D_dt, fem.locate_dofs_topological(self.V, fdim, boundary_facets))

    def source(self, alpha=3, beta=1.2):
        # The source term
        self.f_help = source_term(alpha, beta, self.t)
        self.f = fem.Function(self.V)
        self.f.interpolate(self.f_help)

    def exact(self, alpha=3, beta=1.2):
        # The exact solution
        self.u_exact = exact_solution(alpha, beta, self.t)
        self.u_n = fem.Function(self.V, name = "u_n")
        self.u_n.interpolate(self.u_exact)

    def a(self):
        self.k = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.uh = fem.Function(self.V, name = "uh")

        ### Save the function u_n, as this is the complete solution. uh is only k
        self.xdmf.write_function(self.u_n, self.t)

        
# Butcher Tableau

bt_a = 1.0
bt_b = 1.0
bt_c = 1.0

imp = RK(bt_a, bt_b, bt_c)

# The method and stage number will be the inputs
# The known solution could be adjustable (ex: the order of t)

imp.mesh()
imp.boundary()
imp.source()
imp.exact()



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