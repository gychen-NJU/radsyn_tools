import numpy as np
import vtk
import matplotlib.pyplot as plt
from .need import *
from pyevtk.hl import gridToVTK

def save_to_vtp_surface(coords, values, output_filename):
    # 创建 vtkPoints 对象并添加点数据
    vtk_points = vtk.vtkPoints()
    num_rows, num_cols, _ = coords.shape
    for i in range(num_rows):
        for j in range(num_cols):
            vtk_points.InsertNextPoint(coords[i, j])

    # 创建 vtkPolyData 对象并设置点
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)

    # 创建 vtkFloatArray 对象来存储物理量（标量数据）
    vtk_values = vtk.vtkFloatArray()
    vtk_values.SetName("Values")
    vtk_values.SetNumberOfComponents(1)
    for i in range(num_rows):
        for j in range(num_cols):
            vtk_values.InsertNextValue(values[i, j])

    # 将标量数据添加到 poly_data 中
    poly_data.GetPointData().SetScalars(vtk_values)

    # 创建 vtkCellArray 对象来定义网格面
    vtk_cells = vtk.vtkCellArray()

    # 定义四边形单元（面）的拓扑结构
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i * num_cols + j)
            quad.GetPointIds().SetId(1, i * num_cols + (j + 1))
            quad.GetPointIds().SetId(2, (i + 1) * num_cols + (j + 1))
            quad.GetPointIds().SetId(3, (i + 1) * num_cols + j)
            vtk_cells.InsertNextCell(quad)

    # 将面添加到 poly_data 中
    poly_data.SetPolys(vtk_cells)

    # 使用 vtkXMLPolyDataWriter 写入 .vtp 文件
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(poly_data)
    writer.Write()
    print(f"Data successfully saved to {output_filename}")

def matplotlib_to_plotly(cmap_name, pl_entries=255):
    cmap = plt.get_cmap(cmap_name)
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale

def data2vts(data_ls, name_ls, vts_name='data', **kwargs):
    """
    Convert a list of data arrays to a VTK structured grid file (.vts).

    Parameters:
    -----------
    data_ls : list of numpy.ndarray
        A list of data arrays to be saved. Each array should be either 3D or a tuple of 3D arrays.
    name_ls : list of str
        A list of names corresponding to each data array in `data_ls`.
    vts_name : str, optional
        The name of the VTK file to be saved (default is 'data').
    **kwargs : dict
        Additional keyword arguments:
        - xyz : tuple of numpy.ndarray
            Cartesian coordinates (x, y, z) for the structured grid.
        - rtp : tuple of numpy.ndarray
            Spherical coordinates (r, theta, phi) for the structured grid. If provided, they will be converted to Cartesian coordinates.

    Returns:
    --------
    None
        The function saves the data to a VTK file and does not return any value.

    Raises:
    -------
    TypeError
        If neither `xyz` nor `rtp` is provided.

    Notes:
    ------
    - The function uses `np.gradient` to measure the time taken for the conversion and saving process.
    - If the data array is not 3D, it is assumed to be a tuple of 3D arrays and is converted accordingly.
    - The VTK file is saved using the `gridToVTK` function from the `evtk.hl` module.

    Example:
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 10)
    >>> y = np.linspace(0, 1, 10)
    >>> z = np.linspace(0, 1, 10)
    >>> X, Y, Z = np.meshgrid(x, y, z)
    >>> data1 = np.sin(X) * np.cos(Y) * np.cos(Z)
    >>> data2 = np.cos(X) * np.sin(Y) * np.sin(Z)
    >>> data2vts([data1, data2], ['data1', 'data2'], vts_name='example', xyz=(X, Y, Z))
    Save data to example.vts.
    Time Used: 0.001 min...
    """
    t0 = time.time()
    xyz = kwargs.get('xyz', None)
    rtp = kwargs.get('rtp', None)
    if xyz is None and rtp is None:
        return TypeError('`xyz` or `rtp` should be provided at least one...')
    X,Y,Z = xyz if xyz is not None else rtp2xyz(rtp)
    pointData = dict()
    for name,data in zip(name_ls,data_ls):
        pointData[name]=data if data.ndim==3 else tuple(idata for idata in data)
    gridToVTK(vts_name, X, Y, Z, 
              pointData=pointData)
    print(f'Save data to {vts_name}.vts.\nTime Used: {(time.time()-t0)/60:8.3} min...')