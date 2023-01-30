import meshio
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_mesh(path):
    """
    Read mesh and return vertices + faces using meshio python package
    """
    mesh = meshio.read(path)
    vertices = mesh.points.astype(np.float32)
    faces = mesh.cells[0].data
    return mesh, vertices, faces


def plot_trajectories_on_mesh(vertices, faces, traj_file):
    """
    Plots trajectories overlaid on mesh 
    """
    padding = np.zeros(3)
    shift = -np.min(vertices, axis=0) + padding
    vertices = vertices + shift
    
    trajectories = np.loadtxt(traj_file)
    trajectories = trajectories.reshape(
        (trajectories.shape[0], int(trajectories.shape[1] / 3), 3)
    )
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
#     for i in range(trajectories.shape[1]):
#         ax.plot(
#             trajectories[:, i, 0],
#             trajectories[:, i, 1],
#             trajectories[:, i, 2],
#             alpha=0.5,
#         )

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.ticklabel_format(style="sci", scilimits=(0, 0))
#     fig.tight_layout()
    
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        alpha=0.25,
    )
    
    plt.show()
