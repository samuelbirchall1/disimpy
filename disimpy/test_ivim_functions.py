

import os
import math
import numba
import pickle
import meshio

import numpy as np
from numba import cuda
import numpy.testing as npt
from scipy.stats import normaltest, kstest
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
)

import gradients, simulations, substrates, utils


SEED = 123

def get_variables(
    path, 
    original_mesh_vertices,
    padding=np.zeros(3),):
    """
    Gets vessel skeleton variables and shifts them so that they match the substrate vertices. 
    
    """

    
    vloc = np.load(path + "velocity_location.npy")
    vdir = np.load(path + "velocity_direction.npy")
    seg_length = np.load(path + "seg_length.npy")

    #Shift the corner of the mesh to the origin and then renormalise
    shift = -np.min(original_mesh_vertices, axis=0) + padding
    vdir = vdir + shift
    vloc = vloc + shift 
    vdir = np.linalg.norm(vdir, axis=1)
    mag = mag[:, np.newaxis]
    vdir = np.divide(vdir, mag)

    return vloc, vdir, seg_length

def test_get_nearest_velocity_direction():
    @cuda.jit()
    def test_kernel(positions, vdir):
        thread_id = cuda.grid(1)
        if thread_id >= positions.shape[0]:
            return 

        #Allocate memory
        step = cuda.local.array(3, numba.float64)
        lls = cuda.local.array(3, numba.int64)
        uls = cuda.local.array(3, numba.int64)
        triangle = cuda.local.array((3, 3), numba.float64)
        normal = cuda.local.array(3, numba.float64)
        shifts = cuda.local.array(3, numba.float64)
        temp_r0 = cuda.local.array(3, numba.float64)
        #Get position and generate step 
        r0 = positions[thread_id, :]
        r0 + vdir[thread_id, :]
        return 


    stream = cuda.stream()

    mesh = meshio.read("/content/drive/MyDrive/mresprojectbits/vascular_mesh_22-10-04_21-52-57_r4.ply")
    vertices = mesh.points.astype(np.float32)
    faces = mesh.cells[0].data

    vdir = np.load("/content/drive/MyDrive/mresprojectbits/velocity_direction.npy")
    vloc = np.load("/content/drive/MyDrive/mresprojectbits/velocity_location.npy")
    padding=np.zeros(3)
    shift = -np.min(vertices, axis=0) + padding
    vdir = vdir + shift
    vloc = vloc + shift 
    vdir = np.linalg.norm(vdir, axis=1)
    mag = mag[:, np.newaxis]
    vdir = np.divide(vdir, mag)

    substrate = substrates.mesh(
    vertices,
    faces,
)

    traj_file = "IVIM_traj.txt"
    n_walkers = 100
    seed = 123
    positions = simulations._fill_mesh(n_walkers, substrate, True, seed)
    simulations._write_traj(traj_file, "w", positions)

    vdir_index = []
    for i in range(len(positions)):
        dis = vloc - positions[i]
        dis = np.linalg.norm(dis, axis=1)
        vdir_index.append(np.where(dis==np.amin(np.abs(dis)))[0][0])

    vdir = vdir[vdir_index]
    d_positions = cuda.to_device(positions, stream=stream)
    d_vdir = cuda.to_device(vdir, stream=stream)
    
    test_kernel[1, 128, stream](d_positions, d_vdir)
    stream.synchronize()
    positions = d_positions.copy_to_host(stream=stream)
    print("Finished calculating initial positions")
    simulations._write_traj(traj_file, "a", positions)

