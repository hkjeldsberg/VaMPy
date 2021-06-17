from __future__ import print_function

from pathlib import Path

import numpy as np
from dolfin import *

from postprocessing_common import STRESS, read_command_line

parameters["reorder_dofs_serial"] = False


def compute_wss(case_path, nu, dt, velocity_degree):
    """
    Loads velocity fields from completed CFD simulation,
    and computes and saves the following hemodynamic quantities:
    (1) WSS - Wall shear stress
    (2) TWSSG - Temporal wall shear stress gradient
    (3) OSI - Oscillatory shear index
    (4) RRT - Relative residence time

    Args:
        velocity_degree (int): Finite element degree of velocity
        case_path (Path): Path to results from simulation
        nu (float): Viscosity
        dt (float): Time step of simulation
    """
    # File paths
    case_path = Path(case_path)
    file_path_u = case_path / "u.h5"
    mesh_path = case_path / "mesh.h5"

    # Start post-processing from 2nd cycle using every 10th time step, or 2000 time steps per cycle
    start = 0  # save_data = 5 -> 10000 / 5 = 2000
    step = 2  # save_data = 5 ->    10 / 5 = 2

    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    # Load mesh
    bm = BoundaryMesh(mesh, 'exterior')

    if MPI.rank(MPI.comm_world) == 0:
        print("Define function spaces")
    V_b1 = VectorFunctionSpace(bm, "CG", 1)
    U_b1 = FunctionSpace(bm, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", velocity_degree)

    if MPI.rank(MPI.comm_world) == 0:
        print("Define functions")
    u = Function(V)

    # RRT
    RRT = Function(U_b1)

    # OSI
    OSI = Function(U_b1)

    # WSS_mean
    WSS_mean = Function(V_b1)
    wss_mean = Function(U_b1)

    # WSS_abs
    WSS_abs = Function(U_b1)
    wss_abs = Function(U_b1)

    # TWSSG
    TWSSG = Function(U_b1)
    twssg_ = Function(U_b1)
    twssg = Function(V_b1)
    tau_prev = Function(V_b1)

    stress = STRESS(u, 0.0, nu, mesh)
    dabla = get_dabla_function()

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    file_counter = start
    while True:
        # Read in velocity solution to vector function u
        try:
            f = HDF5File(MPI.comm_world, file_path_u.__str__(), "r")
            vec_name = "/velocity/vector_%d" % file_counter
            timestamp = f.attributes(vec_name)["timestamp"]
            print("=" * 10, "Timestep: {}".format(timestamp), "=" * 10)
            f.read(u, vec_name)
        except:
            print("=" * 10, "Finished reading solutions", "=" * 10)
            break

        # Compute WSS
        if MPI.rank(MPI.comm_world) == 0:
            print("Compute WSS (mean)")
        tau = stress()
        tau.vector()[:] = tau.vector()[:] * 1000
        WSS_mean.vector().axpy(1, tau.vector())

        if MPI.rank(MPI.comm_world) == 0:
            print("Compute WSS (absolute value)")
        dabla(tau.vector(), wss_abs.vector())
        WSS_abs.vector().axpy(1, wss_abs.vector())

        # Compute TWSSG
        if MPI.rank(MPI.comm_world) == 0:
            print("Compute TWSSG")
        twssg.vector().set_local((tau.vector().get_local() - tau_prev.vector().get_local()) / dt)
        twssg.vector().apply("insert")
        dabla(twssg.vector(), twssg_.vector())
        TWSSG.vector().axpy(1, twssg_.vector())

        # Update tau
        if MPI.rank(MPI.comm_world) == 0:
            print("Update WSS \n")
        tau_prev.vector().zero()
        tau_prev.vector().axpy(1, tau.vector())

        # Update file_counter
        file_counter += step

    print("=" * 10, "Saving hemodynamic indices", "=" * 10)
    n = (file_counter - start) // step
    TWSSG.vector()[:] = TWSSG.vector()[:] / n
    WSS_abs.vector()[:] = WSS_abs.vector()[:] / n
    WSS_mean.vector()[:] = WSS_mean.vector()[:] / n

    WSS_abs.rename("WSS", "WSS")
    TWSSG.rename("TWSSG", "TWSSG")

    try:
        dabla(WSS_mean.vector(), wss_mean.vector())
        wss_mean_vec = wss_mean.vector().get_local()
        wss_abs_vec = WSS_abs.vector().get_local()

        # Compute RRT and OSI based on mean and absolute WSS
        RRT.vector().set_local(1 / wss_mean_vec)
        RRT.vector().apply("insert")
        RRT.rename("RRT", "RRT")

        OSI.vector().set_local(0.5 * (1 - wss_mean_vec / wss_abs_vec))
        OSI.vector().apply("insert")
        OSI.rename("OSI", "OSI")
        save = True
    except:
        print("Failed to compute OSI and RRT")
        save = False

    if save:
        # Save OSI and RRT
        rrt_path = (case_path / "RRT.xdmf").__str__()
        osi_path = (case_path / "OSI.xdmf").__str__()

        rrt = XDMFFile(MPI.comm_world, rrt_path)
        osi = XDMFFile(MPI.comm_world, osi_path)

        for f in [rrt, osi]:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            f.parameters["rewrite_function_mesh"] = False

        rrt.write(RRT)
        osi.write(OSI)

    # Save WSS and TWSSG
    wss_path = (case_path / "WSS.xdmf").__str__()
    twssg_path = (case_path / "TWSSG.xdmf").__str__()

    wss = XDMFFile(MPI.comm_world, wss_path)
    twssg = XDMFFile(MPI.comm_world, twssg_path)

    for f in [wss, twssg]:
        f.parameters["flush_output"] = True
        f.parameters["functions_share_mesh"] = True
        f.parameters["rewrite_function_mesh"] = False

    wss.write(WSS_abs)
    twssg.write(TWSSG)


def get_dabla_function():
    """
    Compiles a string in C++ and expose as a Python object (dabla),
    used to compute several hemodynamic quantities.

    Returns:
        dabla: C++ compiled function
    """

    cpp_code = """
    #include <pybind11/pybind11.h>
    #include <dolfin.h>
    namespace dolfin
    {
        void dabla(dolfin::GenericVector& a, dolfin::GenericVector& b) {
            for (unsigned int i=0; i < b.size(); i++) {
                b.setitem(i, pow((pow(a[i], 2) + pow(a[b.size() + i], 2) + pow(a[2 * b.size() + i], 2) ), 0.5));
            }
        }
    }
    PYBIND11_MODULE(SIGNATURE, m)
    {
        m.def("dabla", &dolfin::dabla);
    }
    """

    dabla = compile_cpp_code(cpp_code).dabla

    return dabla


if __name__ == '__main__':
    folder, nu, dt, velocity_degree = read_command_line()
    compute_wss(folder, nu, dt, velocity_degree)
