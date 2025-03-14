from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, apply_lifting, set_bc, create_vector
import ufl
from mpi4py import MPI
import numpy

def rk_solver(method, domain, u_n, u_exact, u_D, bc, dt, num_steps, f, V):
    """
    General RK solver for Euler, Heun, and RK4 methods.

    Parameters:
    - method: Name of the RK method ('Euler', 'Heun', 'RK4').
    - u_n: FEM Function representing the solution at the previous time step.
    - u_exact: Exact solution object.
    - u_D: Dirichlet boundary function.
    - bc: Dirichlet boundary condition.
    - a: Bilinear form (for matrix assembly).
    - L: Linear form (for RHS assembly).
    - solver: PETSc solver.
    - dt: Time step size.
    - num_steps: Number of time steps.
    - alpha: Thermal diffusivity.
    - f: Source term as a constant.
    - V: Function space.
    - v: Test function.
    """
    
    if method == "Backward Euler" or "Explicit Euler":
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        uh = fem.Function(V)

        for n in range(num_steps):
            # We can now create our variational formulation, with the bilinear form `a` and  linear form `L`.
            if method == "Backward Euler":
                F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
            else:
                F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

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
        error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")

        # Compute values at mesh vertices
        error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"Error_max: {error_max:.2e}")
        return
    
    elif method == "Lobatto IIIC":
        # Lobatto IIIC Method (two-stage)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        uh = fem.Function(V)
        k1 = fem.TrialFunction(V)
        k2 = fem.TrialFunction(V)

        for n in range(num_steps):
            ## Stage 1
            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k1.x.petsc_vec)
            k1.x.scatter_forward()

            # Stage 2
            u_temp = fem.Function(V)
            u_temp.x.array[:] = u_n.x.array + dt * k1.x.array
            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_temp), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k2.x.petsc_vec)
            k2.x.scatter_forward()

            # Combine stages
            uh.x.array[:] = u_n.x.array + (dt / 2) * (k1.x.array + k2.x.array)
            u_n.x.array[:] = uh.x.array
    
        V_ex = fem.functionspace(domain, ("Lagrange", 2))
        u_ex = fem.Function(V_ex)
        u_ex.interpolate(u_exact)
        error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")

        # Compute values at mesh vertices
        error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"Error_max: {error_max:.2e}")
        return
    


    elif method == "Heun":
        # Heun's Method (RK2)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        uh = fem.Function(V)
        k1 = fem.Function(V)
        k2 = fem.Function(V)

        for n in range(num_steps):
            ## Stage 1
            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k1.x.petsc_vec)
            k1.x.scatter_forward()

            # Stage 2
            u_temp = fem.Function(V)
            u_temp.x.array[:] = u_n.x.array + dt * k1.x.array
            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_temp), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k2.x.petsc_vec)
            k2.x.scatter_forward()

            # Combine stages
            uh.x.array[:] = u_n.x.array + (dt / 2) * (k1.x.array + k2.x.array)
            u_n.x.array[:] = uh.x.array
    
        V_ex = fem.functionspace(domain, ("Lagrange", 2))
        u_ex = fem.Function(V_ex)
        u_ex.interpolate(u_exact)
        error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")

        # Compute values at mesh vertices
        error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"Error_max: {error_max:.2e}")
        return
    


    elif method == "Radau IIa":
        # Implicit Radau IIa method (two-stage third-order example)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        uh = fem.Function(V)
        k1 = fem.Function(V)
        k2 = fem.Function(V)

        for n in range(num_steps):
            ## Stage 1
            F = k1 * v * ufl.dx + dt * ufl.dot(ufl.grad(k1), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k1.x.petsc_vec)
            k1.x.scatter_forward()

            ## Stage 2
            u_temp = fem.Function(V)
            u_temp.x.array[:] = u_n.x.array + 0.5 * dt * k1.x.array
            F = k2 * v * ufl.dx + dt * ufl.dot(ufl.grad(k2), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k2.x.petsc_vec)
            k2.x.scatter_forward()

            # Combine stages
            uh.x.array[:] = u_n.x.array + (dt / 2) * (k1.x.array + k2.x.array)
            u_n.x.array[:] = uh.x.array
    
        V_ex = fem.functionspace(domain, ("Lagrange", 2))
        u_ex = fem.Function(V_ex)
        u_ex.interpolate(u_exact)
        error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")

        # Compute values at mesh vertices
        error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"Error_max: {error_max:.2e}")
        return






    if method == "RK4":
        # RK4 Method
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        uh = fem.Function(V)

        for n in range(num_steps):
            k1 = fem.Function(V)
            k2 = fem.Function(V)
            k3 = fem.Function(V)
            k4 = fem.Function(V)

            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k1.x.petsc_vec)
            k1.x.scatter_forward()

            ## Stage 2
            u_temp = fem.Function(V)
            u_temp.x.array[:] = u_n.x.array + 0.5 * dt * k1.x.array
            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_temp), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k2.x.petsc_vec)
            k2.x.scatter_forward()

            # Stage 3
            u_temp = fem.Function(V)
            u_temp.x.array[:] = u_n.x.array + 0.5 * dt * k2.x.array
            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_temp), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k3.x.petsc_vec)
            k3.x.scatter_forward()

            # Stage 4
            u_temp = fem.Function(V)
            u_temp.x.array[:] = u_n.x.array + dt * k3.x.array
            F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u_temp), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k4.x.petsc_vec)
            k4.x.scatter_forward()

            # Combine stages
            uh.x.array[:] = u_n.x.array + (dt / 6) * (k1.x.array + 2 * k2.x.array + 2 * k3.x.array + k4.x.array)
            u_n.x.array[:] = uh.x.array
    
        V_ex = fem.functionspace(domain, ("Lagrange", 2))
        u_ex = fem.Function(V_ex)
        u_ex.interpolate(u_exact)
        error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")

        # Compute values at mesh vertices
        error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"Error_max: {error_max:.2e}")
        return
            

    elif method == "Lobatto IIIc":
        # Implicit Lobatto IIIc method (two-stage second-order example)
        k1 = fem.Function(V)
        k2 = fem.Function(V)

        # Solve for the first stage
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, fem.form(k1 * v * ufl.dx - dt * alpha * ufl.dot(ufl.grad(k1), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx))
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        solver.solve(b, k1.x.petsc_vec)
        k1.x.scatter_forward()

        # Solve for the second stage
        u_temp = fem.Function(V)
        u_temp.x.array[:] = u_n.x.array + dt * k1.x.array
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, fem.form(k2 * v * ufl.dx - dt * alpha * ufl.dot(ufl.grad(k2), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx))
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        solver.solve(b, k2.x.petsc_vec)
        k2.x.scatter_forward()

        # Combine stages
        uh.x.array[:] = u_n.x.array + (dt / 2) * (k1.x.array + k2.x.array)
        u_n.x.array[:] = uh.x.array

    elif method == "Gauss-Legendre":
        # Implicit Gauss-Legendre method (two-stage second-order example)
        k1 = fem.Function(V)
        k2 = fem.Function(V)

        # Solve for the first stage
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, fem.form(k1 * v * ufl.dx - dt * alpha * ufl.dot(ufl.grad(k1), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx))
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        solver.solve(b, k1.x.petsc_vec)
        k1.x.scatter_forward()

        # Solve for the second stage
        u_temp = fem.Function(V)
        u_temp.x.array[:] = u_n.x.array + 0.5 * dt * k1.x.array
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, fem.form(k2 * v * ufl.dx - dt * alpha * ufl.dot(ufl.grad(k2), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx))
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        solver.solve(b, k2.x.petsc_vec)
        k2.x.scatter_forward()

        # Combine stages
        uh.x.array[:] = u_n.x.array + (dt / 2) * (k1.x.array + k2.x.array)
        u_n.x.array[:] = uh.x.array

    else:
        raise ValueError(f"Unknown method: {method}. Available methods are: ('Euler', 'Heun', 'RK4', 'Radau IIa', 'Lobatto IIIc', 'Gauss-Legendre')")





"""
OLD RADAU IIA
    elif method == "Radau IIa":
        # Implicit Radau IIa method (two-stage third-order example)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        uh = fem.Function(V)
        k1 = fem.Function(V)
        k2 = fem.Function(V)

        for n in range(num_steps):
            ## Stage 1
            F = k1 * v * ufl.dx + dt * ufl.dot(ufl.grad(k1), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k1.x.petsc_vec)
            k1.x.scatter_forward()

            ## Stage 2
            u_temp = fem.Function(V)
            u_temp.x.array[:] = u_n.x.array + 0.5 * dt * k1.x.array
            F = k2 * v * ufl.dx + dt * ufl.dot(ufl.grad(k2), ufl.grad(v)) * ufl.dx - (u_temp + dt * f) * v * ufl.dx
            a = fem.form(ufl.lhs(F))
            L = fem.form(ufl.rhs(F))

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
            u_D.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, k2.x.petsc_vec)
            k2.x.scatter_forward()

            # Combine stages
            uh.x.array[:] = u_n.x.array + (dt / 2) * (k1.x.array + k2.x.array)
            u_n.x.array[:] = uh.x.array
    
        V_ex = fem.functionspace(domain, ("Lagrange", 2))
        u_ex = fem.Function(V_ex)
        u_ex.interpolate(u_exact)
        error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")

        # Compute values at mesh vertices
        error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
        if domain.comm.rank == 0:
            print(f"Error_max: {error_max:.2e}")
        return

"""