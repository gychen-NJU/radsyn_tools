import numpy as np
import vtk

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
