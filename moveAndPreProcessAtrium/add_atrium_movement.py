import shutil

import matplotlib.pyplot as plt
from morphman.common import *

case_path = "case6.vtp"
case_path = "Case01.vtp"

model = case_path.replace(".vtp", "")
cl_path = model + "_cl.vtp"

# Params
f = 1  # One cycle per second
A = 25 / 2
t_array = np.linspace(0, 1, 20)
volumes = []

# Idea: Compute centerlines and get center of mitral valve as new origin
surface = read_polydata(case_path)

# Capp surface with flow extensions
capped_surface = vmtk_cap_polydata(surface)
inlet, outlets = get_inlet_and_outlet_centers(surface, model)
centerlines, _, _ = compute_centerlines(inlet, outlets, cl_path, capped_surface, resampling=0.5)
centerline = extract_single_line(centerlines, 0)
o = centerline.GetPoint(0)
print(o)


def IdealVolume(x, plot=False, analytical=False):
    if analytical:
        s0 = 0.27
        s1 = 0.15
        x0 = 0.15
        x1 = 0.70
        max_y = 0.71

        A_wave = 10 * (x ** 2 * np.exp(- ((x - x0) / s0) ** 2))  # - (T - t) ** 2 * np.exp(-f * (T - t)))
        E_wave = 0.4 * (np.exp(- ((x - x1) / s1) ** 2))  # - (T - t) ** 2 * np.exp(-f * (T - t)))

        volume = (A_wave + E_wave) / max_y
        if plot:
            t_ = np.linspace(0, 1, len(volume))
            plt.plot(t_, volume)
            plt.show()

    else:
        LA_volume = [36858.89622880263, 42041.397558417586, 47203.72790128924, 51709.730141809414,
                     56494.613640032476, 53466.224048278644, 46739.80937044214, 45723.76234837754,
                     46107.69142568748, 34075.82037837897]

        time = np.linspace(0, 1, len(LA_volume))
        LA_smooth = splrep(time, LA_volume, s=1e6, per=True)
        vmin = 37184.998997815936
        vmax = 19490.21405487303
        volume = (splev(x, LA_smooth) - vmin) / vmax

        if plot:
            time = np.linspace(0, 1, len(LA_volume))
            plt.plot(time, LA_volume)
            print(max(splev(time, LA_smooth)))
            time = np.linspace(0, 1, 1000)
            plt.plot(time, splev(time, LA_smooth), "r-", linewidth=3)
            plt.show()

    return volume


def main():
    for i, t in enumerate(t_array):
        surface = read_polydata(case_path)
        points = surface.GetPoints()

        for j in range(surface.GetNumberOfPoints()):

            p = points.GetPoint(j)
            p = np.asarray(p)

            threshold = -70

            displacement = IdealVolume(t)

            # Axial:
            x_o = o[0]
            y_o = o[1]
            z_o = o[2]
            x_0 = p[0]
            y_0 = p[1]
            z_0 = p[2]

            scaling_x = (x_0 - x_o)  # / abs(x_0 - x_o)
            scaling_y = (y_0 - y_o)  # / abs(y_0 - y_o)
            x_new = A / 100 * scaling_x * displacement
            y_new = A / 100 * scaling_y * displacement

            # Longitudal
            B = 25 / 2 * 2.5

            if z_0 >= 0:
                z_new = B * displacement
            elif 0 > z_0 > threshold:
                scaling_factor = (p[2] / abs(threshold)) + 1
                z_new = scaling_factor * B * displacement
            else:
                z_new = 0

            z_new = B * displacement

            # R = np.asarray([[cost, -sint,0], [sint, cost,0], [0,0,1]])

            p_new = np.asarray([p[0] + x_new, p[1] + y_new, p[2] + z_new])
            points.SetPoint(j, p_new)

        surface.SetPoints(points)

        write_polydata(surface, "moved01/case01_%03d.vtp" % (5 + 5 * i))
        capped_surface = vmtk_cap_polydata(surface)

        volume = vtk_compute_mass_properties(capped_surface, compute_volume=True)
        volumes.append(volume)

    plt.plot(t_array, (np.array(volumes)) * 1E-3, label="Volume (ml)")
    plt.xlabel("time [s]")
    plt.ylabel("LA and LAA volume [ml]")
    plt.show()
    surface = read_polydata(case_path)
    write_polydata(surface, "moved01/case01_000.vtp")

    dest_f = "../flow_extensions"
    dest = "../flow_extensions" + "/moved01"
    if path.isdir(dest):
        shutil.rmtree(dest)

    # shutil.copytree("moved", dest)
    # shutil.copy("moved01/case6.vtp", dest_f)


#t__ = np.linspace(0, 1, 1000)
#IdealVolume(t__, True)
#main()
