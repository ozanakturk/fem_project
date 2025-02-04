"""
Solves the heat equation PDE:

                2         
∂              ∂          
──(u(t, x)) = ───(u(t, x))  +  f
∂t              2         
              ∂x          

in a rod with x ∈ [0.0, 1.0]

with homogeneous Dirichlet Boundary Conditions (u(0.0) = u(1.0) = 0.0)

and a heat source f

using a sine initial condition:

      1 |                      ...........                      
        |                 .....           .....                 
        |              ...                     ...              
        |            ..                           ..            
0.55555 |----------..-------------------------------..----------
        |       ...                                   ...       
        |     ..                                         ..     
        |   ..                                             ..   
        | ..                                                 .. 
      0 |_______________________________________________________
         0                          0.5                        1
"""

import dolfinx as fe
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_elements = 32
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, n_elements)
    #domain = mesh.create_unit_square(MPI.COMM_WORLD, n_elements, n_elements, mesh.CellType.triangle)
    #mesh = mesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, 1.0])


    # Define a Function Space
    lagrange_polynomial_space_first_order = fem.functionspace(
        domain,
        ("Lagrange",
        1)
    )

    # Create boundary condition

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(
        PETSc.ScalarType(0), 
        fem.locate_dofs_topological(
            lagrange_polynomial_space_first_order, 
            fdim, 
            boundary_facets), 
        lagrange_polynomial_space_first_order)

    
    # The initial condition, u(t=0, x) = sin(pi * x)
    """initial_condition = fem.Expression(
        "sin(3.141 * x[0])"
    )"""
    

    def initial_condition(x):
        return np.exp(np.sin(3.141 * x[0]))

    u_old = fem.Function(lagrange_polynomial_space_first_order)
    u_old.name = "u_old"
    u_old.interpolate(initial_condition)

    """# Discretize the initial condition
    u_old = fe.interpolate(
        initial_condition,
        lagrange_polynomial_space_first_order
    )"""

    xdmf = fe.io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(u_old, 0) # t=0

    # The time stepping of the implicit Euler discretization (=dt)
    time_step_length = 0.1

    # The forcing on the rhs of the PDE
    heat_source = fem.Constant(domain, PETSc.ScalarType(0.0))

    # Create the Finite Element Problem
    u_trial = ufl.TrialFunction(lagrange_polynomial_space_first_order)
    v_test = ufl.TestFunction(lagrange_polynomial_space_first_order)

    weak_form_residuum = (
        u_trial * v_test * ufl.dx
        +
        time_step_length * ufl.dot(
            ufl.grad(u_trial),
            ufl.grad(v_test),
        ) * ufl.dx
        -
        (
            u_old * v_test * ufl.dx
            +
            time_step_length * heat_source * v_test * ufl.dx
        )
    )
    
    # We have a linear PDE that is separable into a lhs and rhs
    weak_form_lhs = ufl.lhs(weak_form_residuum)
    weak_form_rhs = ufl.rhs(weak_form_residuum)

    # The function we will be solving for at each point in time
    u_solution = fem.Function(lagrange_polynomial_space_first_order)

    # time stepping
    n_time_steps = 5

    time_current = 0.0
    for i in range(n_time_steps):
        time_current += time_step_length

        # Finite Element Assembly, BC imprint & solving the linear system
        ufl.solve(
            weak_form_lhs == weak_form_rhs,
            u_solution,
            boundary_condition,
        )

        u_old.assign(u_solution)

        fe.plot(u_solution, label=f"t={time_current:1.1f}")
    
    plt.legend()
    plt.title("Heat Conduction in a rod with homogeneous Dirichlet BC")
    plt.xlabel("x position")
    plt.ylabel("Temperature")
    plt.grid()
    plt.show()

