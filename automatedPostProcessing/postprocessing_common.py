from argparse import ArgumentParser

try:
    from dolfin import *
    try:
        parameters["allow_extrapolation"] = True
    except NameError:
        pass
except ImportError:
    pass


def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--case', type=str, default="/results_folder/1/VTK", help="Path to simulation results",
                        metavar="PATH")
    parser.add_argument('--nu', type=float, default=3.3018e-3, help="Viscosity used in simulation")
    parser.add_argument('--dt', type=float, default=0.0951, help="Time step of simulation")
    parser.add_argument('--velocity-degree', type=int, default=1, help="Degree of velocity element")
    parser.add_argument('--no-of-cycles', type=float, default=1, help="Number of cardiac cycles")

    args = parser.parse_args()

    return args.case, args.nu, args.dt, args.velocity_degree, args.no_of_cycles


def epsilon(u):
    """
    Computes the strain-rate tensor
    Args:
        u (Function): Velocity field

    Returns:
        epsilon (Function): Strain rate tensor of u
    """

    return 0.5 * (grad(u) + grad(u).T)


class STRESS:
    def __init__(self, u, p, nu, mesh):
        boundary_mesh = BoundaryMesh(mesh, 'exterior')
        self.bmV = VectorFunctionSpace(boundary_mesh, 'CG', 1)

        # Compute stress tensor
        sigma = (2 * nu * epsilon(u)) - (p * Identity(len(u)))

        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        Ft = F - (Fn * n)  # vector-valued

        # Integrate against piecewise constants on the boundary
        scalar = FunctionSpace(mesh, 'DG', 0)
        vector = VectorFunctionSpace(mesh, 'CG', 1)
        scaling = FacetArea(mesh)  # Normalise the computed stress relative to the size of the element

        v = TestFunction(scalar)
        w = TestFunction(vector)

        # Create functions
        self.Fn = Function(scalar)
        self.Ftv = Function(vector)
        self.Ft = Function(scalar)

        self.Ln = 1 / scaling * v * Fn * ds
        self.Ltv = 1 / (2 * scaling) * inner(w, Ft) * ds
        self.Lt = 1 / scaling * inner(v, self.norm_l2(self.Ftv)) * ds

    def __call__(self):
        """
        Compute stress for given velocity field u and pressure field p

        Returns:
            Ftv_mb (Function): Shear stress
        """

        # Assemble vectors
        assemble(self.Ltv, tensor=self.Ftv.vector())
        self.Ftv_bm = interpolate(self.Ftv, self.bmV)

        return self.Ftv_bm

    def norm_l2(self, u):
        """
        Compute norm of vector u in expression form
        Args:
            u (Function): Function to compute norm of

        Returns:
            norm (Power): Norm as expression
        """
        return pow(inner(u, u), 0.5)
