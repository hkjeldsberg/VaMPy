##   Copyright (c) Aslak W. Bergersen, Henrik A. Kjeldsberg. All rights reserved.
##   See LICENSE file for details.

##      This software is distributed WITHOUT ANY WARRANTY; without even
##      the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##      PURPOSE.  See the above copyright notices for more information.
import math
import sys

import numpy as np
import vtk

radiusArrayName = 'MaximumInscribedSphereRadius'


def get_cell_data_array(array_name, line, k=1):
    """
    Get data array from polydata object (CellData).

    Args:
        array_name (str): Name of array.
        line (vtkPolyData): Centerline object.
        k (int): Dimension.

    Returns:
        array (ndarray): Array containing data points.
    """
    # w numpy support, n=63000 32.7 ms, wo numpy support 33.4 ms
    # vtk_array = line.GetCellData().GetArray(arrayName)
    # array = numpy_support.vtk_to_numpy(vtk_array)
    array = np.zeros((line.GetNumberOfCells(), k))
    if k == 1:
        data_array = line.GetCellData().GetArray(array_name).GetTuple1
    elif k == 2:
        data_array = line.GetCellData().GetArray(array_name).GetTuple2
    elif k == 3:
        data_array = line.GetCellData().GetArray(array_name).GetTuple3
    elif k == 9:
        data_array = line.GetCellData().GetArray(array_name).GetTuple9

    for i in range(line.GetNumberOfCells()):
        array[i, :] = data_array(i)

    return array


def get_point_data_array(array_name, line, k=1):
    """
    Get data array from polydata object (PointData).

    Args:
        array_name (str): Name of array.
        line (vtkPolyData): Centerline object.
        k (int): Dimension.

    Returns:
        array (ndarray): Array containing data points.
    """
    # w numpy support, n=63000 32.7 ms, wo numpy support 33.4 ms
    # vtk_array = line.GetPointData().GetArray(arrayName)
    # array = numpy_support.vtk_to_numpy(vtk_array)
    array = np.zeros((line.GetNumberOfPoints(), k))
    if k == 1:
        data_array = line.GetPointData().GetArray(array_name).GetTuple1
    elif k == 2:
        data_array = line.GetPointData().GetArray(array_name).GetTuple2
    elif k == 3:
        data_array = line.GetPointData().GetArray(array_name).GetTuple3
    elif k == 9:
        data_array = line.GetPointData().GetArray(array_name).GetTuple9

    for i in range(line.GetNumberOfPoints()):
        array[i, :] = data_array(i)

    return array


def vtk_compute_connectivity(surface, mode="All", closest_point=None, show_color_regions=True,
                             mark_visited_points=False):
    """Wrapper of vtkPolyDataConnectivityFilter. Compute connectivity.

    Args:
        show_color_regions (bool): Turn on/off the coloring of connected regions.
        mark_visited_points (bool): Specify whether to record input point ids that appear in the output.
        surface (vtkPolyData): Input surface data.
        mode (str): Type of connectivity filter.
        closest_point (list): Point to be used for mode='Closest'
    """
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(surface)

    # Mark each region with "RegionId"
    if mode == "All":
        connectivity.SetExtractionModeToAllRegions()
    elif mode == "Largest":
        connectivity.SetExtractionModeToLargestRegion()
    elif mode == "Closest":
        if closest_point is None:
            print("ERROR: point not set for extracting closest region")
            sys.exit(0)
        connectivity.SetExtractionModeToClosestPointRegion()
        connectivity.SetClosestPoint(closest_point)

    if show_color_regions:
        connectivity.ColorRegionsOn()

    if mark_visited_points:
        connectivity.MarkVisitedPointIdsOn()

    connectivity.Update()
    output = connectivity.GetOutput()

    return output


def vtk_convert_unstructured_grid_to_polydata(unstructured_grid):
    """Wrapper for vtkGeometryFilter, which converts an unstructured grid into a polydata.

    Args:
        unstructured_grid (vtkUnstructuredGrid): An unstructured grid.

    Returns:
        surface (vtkPolyData): A vtkPolyData object from the unstrutured grid.
    """
    # Convert unstructured grid to polydata
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(unstructured_grid)
    geo_filter.Update()
    polydata = geo_filter.GetOutput()

    return polydata


def vtk_compute_threshold(surface, name, lower=0, upper=1, threshold_type="between", source=1):
    """Wrapper for vtkThreshold. Extract a section of a surface given a criteria.

    Args:
        surface (vtkPolyData): The input data to be extracted.
        name (str): Name of scalar array.
        lower (float): Lower bound.
        upper (float): Upper bound.
        threshold_type (str): Type of threshold (lower, upper, between)
        source (int): PointData or CellData.

    Returns:
        surface (vtkPolyData): The extracted surface based on the lower and upper limit.
    """
    # source = 1 uses cell data as input
    # source = 0 uses point data as input

    # Apply threshold
    vtk_threshold = vtk.vtkThreshold()
    vtk_threshold.SetInputData(surface)
    if threshold_type == "between":
        vtk_threshold.ThresholdBetween(lower, upper)
    elif threshold_type == "lower":
        vtk_threshold.ThresholdByLower(lower)
    elif threshold_type == "upper":
        vtk_threshold.ThresholdByUpper(upper)
    else:
        print((("%s is not a threshold type. Pleace chose from: upper, lower" +
                ", or between") % threshold_type))
        sys.exit(0)

    vtk_threshold.SetInputArrayToProcess(0, 0, 0, source, name)
    vtk_threshold.Update()
    surface = vtk_threshold.GetOutput()

    # Convert to polydata
    surface = vtk_convert_unstructured_grid_to_polydata(surface)

    return surface


def vtk_extract_feature_edges(polydata, compute_feature_edges=False, compute_boundary_edges=True,
                              compute_non_manifold_edges=False):
    """Wrapper for vtkFeatureedges. Extracts the edges of the cells that are open.

    Args:
        compute_non_manifold_edges (bool): Turn on/off the extraction of non-manifold edges.
        compute_boundary_edges (bool): Turn on/off the extraction of boundary edges.
        compute_feature_edges (bool): Turn on/off the extraction of feature edges.
        polydata (vtkPolyData): surface to extract the openings from.

    Returns:
        feature_edges (vtkPolyData): The boundary edges of the surface.
    """
    feature_edges = vtk.vtkFeatureEdges()
    if compute_feature_edges:
        feature_edges.FeatureEdgesOn()
    else:
        feature_edges.FeatureEdgesOff()
    if compute_boundary_edges:
        feature_edges.BoundaryEdgesOn()
    else:
        feature_edges.BoundaryEdgesOff()
    if compute_non_manifold_edges:
        feature_edges.NonManifoldEdgesOn()
    else:
        feature_edges.NonManifoldEdgesOff()
    feature_edges.SetInputData(polydata)
    feature_edges.Update()

    return feature_edges.GetOutput()


def get_vtk_cell_locator(surface):
    """Wrapper for vtkCellLocator

    Args:
        surface (vtkPolyData): input surface

    Returns:
        return (vtkCellLocator): Cell locator of the input surface.
    """
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    return locator


def vtk_compute_mass_properties(surface, compute_surface_area=True, compute_volume=False):
    """
    Calculate the volume from the given polydata

    Args:
        compute_volume (bool): Compute surface volume if True
        compute_surface_area (bool): Compute surface area if True
        surface (vtkPolyData): Surface to compute are off

    Returns:
        area (float): Area of the input surface
    Returns:
        volume (float): Volume of the input surface
    """
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)

    if compute_surface_area:
        return mass.GetSurfaceArea()

    if compute_volume:
        return mass.GetVolume()


def vtk_compute_normal_gradients(cell_normals, use_faster_approximation=False):
    """
    Compute gradients of the normals

    Args:
        cell_normals (vtkPolyData): Surface to compute normals on
        use_faster_approximation (bool): Use a less accurate algorithm that performs fewer calculations, but faster.
    """
    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(cell_normals)
    gradient_filter.SetInputArrayToProcess(0, 0, 0, 1, "Normals")
    if use_faster_approximation:
        gradient_filter.FasterApproximationOn()

    gradient_filter.Update()
    gradients = gradient_filter.GetOutput()

    return gradients


def vtk_compute_polydata_normals(surface, compute_point_normals=False, compute_cell_normals=False):
    """ Wrapper for vtkPolyDataNormals

    Args:
        surface (vtkPolyData): Surface model
        compute_point_normals (bool): Turn on/off the computation of point normals.
        compute_cell_normals (bool): Turn on/off the computation of cell normals.

    Returns:
        vtkPolyData: Cell normals of surface model
    """
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(surface)
    if compute_point_normals:
        normal_generator.ComputePointNormalsOn()
    else:
        normal_generator.ComputePointNormalsOff()
    if compute_cell_normals:
        normal_generator.ComputeCellNormalsOn()
    else:
        normal_generator.ComputeCellNormalsOff()

    normal_generator.Update()
    cell_normals = normal_generator.GetOutput()

    return cell_normals


def compute_circleness(surface):
    """Compute the area ratio betwen minimum circle and the maximum circle.

    Args:
        surface (vtkPolyData): Boundary edges of an opening

    Returns:
        circleness (float): Area ratio
        center (list): Center of the opening.
    """
    edges = vtk_extract_feature_edges(surface)

    # Get points
    points = []
    for i in range(edges.GetNumberOfPoints()):
        points.append(edges.GetPoint(i))

    # Compute center
    points = np.array(points)
    center = np.mean(np.array(points), axis=0)

    # Compute ratio between max inscribed sphere, and min inscribed "area"
    point_radius = np.sqrt(np.sum((points - center) ** 2, axis=1))
    argsort = np.argsort(point_radius)
    if point_radius[argsort[1]] / point_radius[argsort[0]] > 5:
        radius_min = point_radius[argsort[1]]
    else:
        radius_min = point_radius.min()

    min_area = math.pi * radius_min ** 2
    max_area = math.pi * point_radius.max() ** 2
    circleness = max_area / min_area

    return circleness, center
