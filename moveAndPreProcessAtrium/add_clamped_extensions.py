import argparse
import os

from IPython import embed
from morphman.common import *
from vtk.numpy_interface import dataset_adapter as dsa

# Global parameters
cell_id_name = "CellEntityIds"


def vmtkSmoother(surface, method, iterations=600):
    """
    Wrapper for a vmtksurfacesmoothing.

    Args:
        surface (vtkPolyData): Input surface to be smoothed.
        method (str): Smoothing method.
        iterations (int): Number of iterations.

    Returns:
        surface (vtkPolyData): The smoothed surface.
    """

    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.NumberOfIterations = iterations
    smoother.Method = method
    smoother.Execute()
    surface = smoother.Surface

    return surface


def add_first_flow_extension(surface, centerlines):
    # Mimick behaviour of vmtkflowextensionfilter
    boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    boundaryExtractor.SetInputData(surface)
    boundaryExtractor.Update()
    boundaries = boundaryExtractor.GetOutput()

    # Find inlet
    lengths = []
    for i in range(boundaries.GetNumberOfCells()):
        lengths.append(get_curvilinear_coordinate(boundaries.GetCell(i))[-1])
    inlet_id = lengths.index(max(lengths))

    # Exclude inlet
    boundaryIds = vtk.vtkIdList()
    for i in range(centerlines.GetNumberOfLines() + 1):
        if i != inlet_id:
            boundaryIds.InsertNextId(i)

    # Extrude the openings
    AdaptiveExtensionLength = 1
    AdaptiveExtensionRadius = 1
    AdaptiveNumberOfBoundaryPoints = 1
    ExtensionLength = 1.0
    ExtensionRatio = 3
    ExtensionRadius = 1.0
    TransitionRatio = 1
    TargetNumberOfBoundaryPoints = 50
    CenterlineNormalEstimationDistanceRatio = 1.0
    ExtensionMode = "centerlinedirection"
    # ExtensionMode = "boundarynormal"
    InterpolationMode = "thinplatespline"
    Sigma = 1.0

    flowExtensionsFilter = vtkvmtk.vtkvmtkPolyDataFlowExtensionsFilter()
    flowExtensionsFilter.SetInputData(surface)
    flowExtensionsFilter.SetCenterlines(centerlines)
    flowExtensionsFilter.SetSigma(Sigma)
    flowExtensionsFilter.SetAdaptiveExtensionLength(AdaptiveExtensionLength)
    flowExtensionsFilter.SetAdaptiveExtensionRadius(AdaptiveExtensionRadius)
    flowExtensionsFilter.SetAdaptiveNumberOfBoundaryPoints(AdaptiveNumberOfBoundaryPoints)
    flowExtensionsFilter.SetExtensionLength(ExtensionLength)
    flowExtensionsFilter.SetExtensionRatio(ExtensionRatio)
    flowExtensionsFilter.SetExtensionRadius(ExtensionRadius)
    flowExtensionsFilter.SetTransitionRatio(TransitionRatio)
    flowExtensionsFilter.SetCenterlineNormalEstimationDistanceRatio(CenterlineNormalEstimationDistanceRatio)
    flowExtensionsFilter.SetNumberOfBoundaryPoints(TargetNumberOfBoundaryPoints)
    if ExtensionMode == "centerlinedirection":
        flowExtensionsFilter.SetExtensionModeToUseCenterlineDirection()
    elif ExtensionMode == "boundarynormal":
        flowExtensionsFilter.SetExtensionModeToUseNormalToBoundary()
    if InterpolationMode == "linear":
        flowExtensionsFilter.SetInterpolationModeToLinear()
    elif InterpolationMode == "thinplatespline":
        flowExtensionsFilter.SetInterpolationModeToThinPlateSpline()
    flowExtensionsFilter.SetBoundaryIds(boundaryIds)
    flowExtensionsFilter.Update()

    surface = flowExtensionsFilter.GetOutput()

    # Smooth at edges
    surface = vmtkSmoother(surface, "laplace", iterations=200)

    return surface


def remesh_surface(surface, edge_length, exclude=None):
    surface = dsa.WrapDataObject(surface)
    if cell_id_name not in surface.CellData.keys():
        surface.CellData.append(np.zeros(surface.VTKObject.GetNumberOfCells()) + 1, cell_id_name)
    remeshing = vmtkscripts.vmtkSurfaceRemeshing()
    remeshing.Surface = surface.VTKObject
    remeshing.CellEntityIdsArrayName = cell_id_name
    remeshing.TargetEdgeLength = edge_length
    remeshing.MaxEdgeLength = 1e6
    remeshing.MinEdgeLength = 0.0
    remeshing.TargetEdgeLengthFactor = 1.0
    remeshing.TargetEdgeLengthArrayName = ""
    remeshing.TriangleSplitFactor = 5.0
    remeshing.ElementSizeMode = "edgelength"
    if exclude is not None:
        remeshing.ExcludeEntityIds = exclude

    remeshing.Execute()

    remeshed_surface = remeshing.Surface

    return remeshed_surface


def move_surface(surface, original, remeshed, remeshed_extended, distance,
                 point_map, file_path, i, points):
    surface = dsa.WrapDataObject(surface)
    original = dsa.WrapDataObject(original)
    remeshed = dsa.WrapDataObject(remeshed)
    remeshed_extended = dsa.WrapDataObject(remeshed_extended)

    if "displacement" in original.PointData.keys():
        original.VTKObject.GetPointData().RemoveArray("displacement")

    if "displacement" in remeshed_extended.PointData.keys():
        remeshed_extended.VTKObject.GetPointData().RemoveArray("displacement")

    # Get displacement field
    original.PointData.append(surface.Points - original.Points, "displacement")

    # Get
    projector = vmtkscripts.vmtkSurfaceProjection()
    projector.Surface = remeshed_extended.VTKObject
    projector.ReferenceSurface = original.VTKObject
    projector.Execute()

    # New surface
    new_surface = projector.Surface
    new_surface = dsa.WrapDataObject(new_surface)

    # Manipulate displacement in the extensions
    displacement = new_surface.PointData["displacement"]
    displacement[remeshed.Points.shape[0]:] = distance * displacement[point_map]

    # Move the mesh points
    new_surface.Points += displacement
    write_polydata(new_surface.VTKObject, file_path)
    points[:, :, i] = new_surface.Points.copy()
    new_surface.Points -= displacement


def get_point_map(remeshed, remeshed_extended):
    remeshed = dsa.WrapDataObject(remeshed)
    remeshed_extended = dsa.WrapDataObject(remeshed_extended)

    # Get lengths
    num_re = remeshed.Points.shape[0]
    num_ext = remeshed_extended.Points.shape[0] - remeshed.Points.shape[0]

    # Get locators
    inner_feature = vtk_compute_connectivity(vtk_extract_feature_edges(remeshed.VTKObject))
    outer_feature = vtk_compute_connectivity(vtk_extract_feature_edges(remeshed_extended.VTKObject))
    locator_inner = get_vtk_point_locator(inner_feature)
    locator_outer = get_vtk_point_locator(outer_feature)
    locator_remeshed = get_vtk_point_locator(remeshed.VTKObject)

    # Wrap objects
    region_inner = dsa.WrapDataObject(inner_feature)
    region_outer = dsa.WrapDataObject(outer_feature)

    # Get distance and point map
    distances = np.zeros(num_ext)
    point_map = np.zeros(num_ext)
    for i in range(num_ext):
        point = remeshed_extended.Points[num_re + i]
        inner_id = locator_inner.FindClosestPoint(point)
        outer_id = locator_outer.FindClosestPoint(point)

        distances[i] = get_distance(point, region_outer.Points[outer_id]) \
                       / get_distance(region_inner.Points[inner_id], region_outer.Points[outer_id])**2
        point_map[i] = locator_remeshed.FindClosestPoint(region_inner.Points[inner_id])

    # Let the points corresponding to the caps have distance 0
    point_map = point_map.astype(int)
    return distances, point_map


def capp_surface(remeshed_extended, remove_inlet=False, offset=1):
    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = remeshed_extended
    capper.Interactive = 0
    capper.Method = "centerpoint"
    capper.TriangleOutput = 0
    capper.CellEntityIdOffset = offset
    capper.Execute()
    surface = capper.Surface

    # Find mitral opening
    if remove_inlet:
        surface = vtk_triangulate_surface(surface)
        surface = vtk_clean_polydata(surface)
        number_of_openings = get_array_cell(cell_id_name, surface)
        area = []
        for i in range(2, int(number_of_openings.max()) + 1):
            region = threshold(surface, cell_id_name, lower=i - 0.1, upper=i + 0.1)
            area.append(compute_area(region))

        # Remove mitral cap
        surface = dsa.WrapDataObject(surface)
        inlet_id = area.index(max(area)) + 2
        ids = surface.CellData[cell_id_name].copy()
        ids[ids == inlet_id] = -1
        ids[ids != -1] = 1
        surface.CellData.append(ids, "threshold")
        surface = threshold(surface.VTKObject, "threshold", lower=0, upper=1e6)

    return surface



def add_flow_extensions(case, extension):
    global remeshed_extended
    # TODO: Add ArgeParse
    # TODO: Move the below in to a main
    # TODO: Consider using a mean center / centerline to set normal of outlets.
    # Create folder with result files
    # File paths
    original_path = case
    original = read_polydata(original_path)
    write_polydata(original, "Case00.vtp")
    model = original_path.replace(".vtp", "")
    moved_path = model + "_moved"
    results_path = model + extension
    extended_path = path.join(results_path, "extended")
    if not path.exists(results_path):
        os.mkdir(results_path)
    if not path.exists(extended_path):
        os.mkdir(extended_path)
    tmp_path = path.join(results_path, model)

    centerline_path = tmp_path + "_cl.vtp"
    points_path = tmp_path + "_points.np"
    mesh_path = tmp_path + ".vtu"
    mesh_xml_path = mesh_path.replace(".vtu", ".xml")
    remeshed_path = tmp_path + "_remeshed.vtp"
    remeshed_extended_path = tmp_path + "_remeshed_extended.vtp"
    base_path = path.join(moved_path, model + "_%03d.vtp")

    # Remesh segmented surface,  Surface mesh resolution
    resolution = 1.9

    if is_surface_capped(original):
        uncapped = get_uncapped_surface(original, area_limit=20, circleness_limit=20, gradients_limit=0.08)
        clipped_path = "clipped.vtp"
        write_polydata(uncapped, clipped_path)

    # remeshed = original
    if path.exists(remeshed_path):
        remeshed = read_polydata(remeshed_path)
    else:
        remeshed = remesh_surface(original, resolution)
        remeshed = vtk_clean_polydata(remeshed)
        write_polydata(remeshed, remeshed_path)

    # Compute centerline
    if not path.exists(centerline_path):
        inlet, outlet = compute_centers(remeshed, tmp_path)
        centerline, _, _ = compute_centerlines(inlet, outlet, centerline_path, capp_surface(remeshed), resampling=0.1,
                                               end_point=1)
    else:
        centerline = read_polydata(centerline_path)

    # Create surface extensions on the original surface
    remeshed_extended = add_first_flow_extension(remeshed, centerline)
    write_polydata(remeshed_extended, remeshed_extended_path)
    # Get a point mapper
    distance, point_map = get_point_map(remeshed, remeshed_extended)
    # Add extents to all surfaces
    points = np.zeros((remeshed_extended.GetNumberOfPoints(), 3, 21))
    for i in range(21):
        tmp_path = base_path % (5 * i)
        if tmp_path == original_path:
            points[:, :, i] = dsa.WrapDataObject(remeshed_extended).Points
            continue

        tmp_surface = read_polydata(tmp_path)
        new_path = path.join(extended_path, tmp_path.split("/")[-1].replace("_", "_extended_"))
        move_surface(tmp_surface, original, remeshed, remeshed_extended, distance,
                     point_map, new_path, i, points)

    points[:, :, -1] = points[:, :, 0]
    points = points  # convert to m
    points.dump(points_path)
    N = 200
    time = np.linspace(0, 1, points.shape[2])
    N2 = N + N // (time.shape[0] - 1)
    move = np.zeros((points.shape[0], points.shape[1], N + 1))
    move[:, 0, :] = resample(points[:, 0, :], N2, time, axis=1)[0][:, :N - N2 + 1]
    move[:, 1, :] = resample(points[:, 1, :], N2, time, axis=1)[0][:, :N - N2 + 1]
    move[:, 2, :] = resample(points[:, 2, :], N2, time, axis=1)[0][:, :N - N2 + 1]
    # move.dump(points_path)
    points.dump(points_path)

    # Cap mitral valve
    exit()
    if not path.exists(mesh_path):
        remeshed_extended = dsa.WrapDataObject(remeshed_extended)
        remeshed_extended.CellData.append(np.zeros(remeshed_extended.VTKObject.GetNumberOfCells()) + 1, cell_id_name)
        # remeshed_extended.Points = remeshed_extended.Points / 1000
        remeshed_all_capped = capp_surface(remeshed_extended.VTKObject)
        remeshed_all_capped = remesh_surface(remeshed_all_capped, resolution, exclude=[1])

        # Mesh volumetric
        sizingFunction = vtkvmtk.vtkvmtkPolyDataSizingFunction()
        sizingFunction.SetInputData(remeshed_all_capped)
        sizingFunction.SetSizingFunctionArrayName("Volume")
        sizingFunction.SetScaleFactor(0.8)
        sizingFunction.Update()

        surfaceToMesh = vmtkscripts.vmtkSurfaceToMesh()
        surfaceToMesh.Surface = sizingFunction.GetOutput()
        surfaceToMesh.Execute()

        tetgen = vmtkscripts.vmtkTetGen()
        tetgen.Mesh = surfaceToMesh.Mesh
        tetgen.GenerateCaps = 0
        tetgen.UseSizingFunction = 1
        tetgen.SizingFunctionArrayName = "Volume"
        tetgen.CellEntityIdsArrayName = cell_id_name
        tetgen.Order = 1
        tetgen.Quality = 1
        tetgen.PLC = 1
        tetgen.NoBoundarySplit = 1
        tetgen.RemoveSliver = 1
        tetgen.OutputSurfaceElements = 1
        tetgen.OutputVolumeElements = 1
        tetgen.Execute()

        Mesh = tetgen.Mesh
        write_polydata(Mesh, mesh_path)

        meshWriter = vmtkscripts.vmtkMeshWriter()
        meshWriter.CellEntityIdsArrayName = "CellEntityIds"
        meshWriter.Mesh = Mesh
        meshWriter.Mode = "ascii"
        meshWriter.Compressed = True
        meshWriter.OutputFileName = mesh_xml_path
        meshWriter.Execute()
        # polyDataVolMesh = mesh


if __name__ == "__main__":
    add_flow_extensions(**read_command_line())
