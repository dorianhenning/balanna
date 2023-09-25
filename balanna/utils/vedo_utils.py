import numpy as np
import trimesh
import trimesh.visual
import vedo
import vtk

from typing import Dict, List, Optional, Tuple
from vtk.util.numpy_support import numpy_to_vtk


def trimesh_scene_2_vedo(scene: trimesh.Scene, label: Optional[str] = None, use_scene_cam: bool = False
                         ) -> Tuple[List[vedo.Mesh], Optional[Dict[str, np.ndarray]]]:
    meshes_vedo = []
    for m in scene.geometry.values():
        if isinstance(m, trimesh.Trimesh):
            m_vedo = vedo.trimesh2vedo(m)
            if m.visual.kind == "vertex":
                face_colors = trimesh.visual.color.vertex_to_face_color(m.visual.vertex_colors, m.faces)
                m_vedo.cellIndividualColors(face_colors)
            meshes_vedo.append(m_vedo)

        elif isinstance(m, trimesh.PointCloud):
            vertices = m.vertices
            vertex_colors = m.visual.vertex_colors
            n, _ = vertices.shape

            # vedo.Points uses vtk.vtkPolyData() as backend data storage and thus converts
            # the input to this type. However, the vedo conversion is quite inefficient, therefore
            # we convert the trimesh.PointCloud before passing it to vedo.
            # partially from https://github.com/pyvista/utilities/helpers.py
            vtkpts = vtk.vtkPoints()
            vtk_arr = numpy_to_vtk(vertices, deep=True)
            vtkpts.SetData(vtk_arr)
            pd = vtk.vtkPolyData()
            pd.SetPoints(vtkpts)

            # For some reason, vedo requires each vtk.vtkPolyData object to set an internal
            # cell array, as it uses the vertices, not the points attribute. As a point cloud
            # is unconnected, the offset and connectivity is just the index of the respective point.
            carr = vtk.vtkCellArray()
            carr.SetData(vedo.numpy2vtk(np.arange(n + 1), dtype="int"),  # offset
                         vedo.numpy2vtk(np.arange(n), dtype="int"))  # connectivity
            pd.SetVerts(carr)

            # Set vertex color RGB/RGBA values as active scalar property of the vtk.vtkPolyData.
            if vertex_colors.shape[1] == 3:
                ucols = numpy_to_vtk(vertex_colors)
                ucols.SetName("Points_RGB")
                pd.GetPointData().AddArray(ucols)
                pd.GetPointData().SetActiveScalars("Points_RGB")
            elif vertex_colors.shape[1] == 4:
                ucols = numpy_to_vtk(vertex_colors)
                ucols.SetName("Points_RGBA")
                pd.GetPointData().AddArray(ucols)
                pd.GetPointData().SetActiveScalars("Points_RGBA")
            else:
                print("\033[93m" + f"Invalid point cloud colors, skipping ..." + "\033[0m")

            # Extract the point size from the trimesh.PointCloud metadata, if available.
            point_size = 4
            if "point_size" in m.metadata:
                point_size = m.metadata["point_size"]

            m_vedo = vedo.Points(pd, r=point_size)
            meshes_vedo.append(m_vedo)

        elif isinstance(m, trimesh.path.Path3D):
            # The trimesh path consists of entities and vertices. The vertices are the 3D points,
            # that are connected as described in the entities.
            if not all([isinstance(me, trimesh.path.entities.Line) for me in m.entities]):
                raise ValueError("Currently only trimesh.path.entities.Line entities are supported")
            if not all([len(me.points) == 2 for me in m.entities]):
                raise ValueError("Invalid line entities, should have point lists [start, end]")

            # Add each line segment individually as a vedo line to support multicolored lines
            # and different alpha values along the line.
            for ke, line_entity in enumerate(m.entities):
                i, j = line_entity.points
                c = m.colors[ke, :3]
                alpha = m.colors[ke, -1] / 255  # [0, 255] -> [0, 1]
                m_vedo = vedo.Lines(m.vertices[None, i], m.vertices[None, j], lw=2, c=c, alpha=alpha)
                meshes_vedo.append(m_vedo)

        else:
            meshes_vedo.append(m)

    if label is not None:
        annotation = vedo.CornerAnnotation()
        annotation.text(label)
        meshes_vedo.append(annotation)

    camera_dict = None
    if use_scene_cam:
        focal_distance = 1.0
        T_W_C = scene.camera_transform
        cam_0 = T_W_C[:3, 3]
        cam_1 = cam_0 + T_W_C[:3, :3] @ np.array([0, 0, focal_distance])  # along z of camera
        view_up = - T_W_C[:3, 1]  # camera convention -> y points down
        camera_dict = dict(pos=cam_0, focal_point=cam_1, viewup=view_up)

    return meshes_vedo, camera_dict
