

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
    mag = np.linalg.norm(vdir, axis=1)
    mag = mag[:, np.newaxis]
    vdir = np.divide(vdir, mag)

    return vloc, vdir, seg_length

def test_flow_simulation():
    @cuda.jit()
    def test_kernel(positions, vdir, step_l):
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
        step = vdir[thread_id, :]
        
        for i in range(3):
          positions[thread_id, i] = r0[i] + step[i]*step_l
        return 


    stream = cuda.stream()

    mesh = meshio.read("/content/drive/MyDrive/mresprojectbits/vascular_mesh_22-10-04_21-52-57_r4.ply")
    vertices = mesh.points.astype(np.float32)*1e-6
    faces = mesh.cells[0].data

    vdir = np.load("/content/drive/MyDrive/mresprojectbits/velocity_direction.npy")
    vloc = np.load("/content/drive/MyDrive/mresprojectbits/velocity_location.npy")
    vloc = vloc*1e-6
    padding=np.zeros(3)
    shift = -np.min(vertices, axis=0) + padding
    vdir = vdir + shift
    vloc = vloc + shift 
    mag = np.linalg.norm(vdir, axis=1)
    mag = mag[:, np.newaxis]
    vdir = np.divide(vdir, mag)

    substrate = substrates.mesh(
    vertices,
    faces,
    periodic=True
)

    traj_file = "IVIM_traj.txt" 
    n_walkers = int(1e3)
    seed = 123
    positions = simulations._fill_mesh(n_walkers, substrate, True, seed)
    simulations._write_traj(traj_file, "w", positions)

    step_l = 1e-5


    vdir_index = []
    for i in range(len(positions)):
        dis = vloc - positions[i]
        dis = np.linalg.norm(dis, axis=1)
        vdir_index.append(np.where(dis==np.amin(np.abs(dis)))[0][0])

    vdir = vdir[vdir_index]
    print(vdir.shape)
    print(positions.shape)
    d_positions = cuda.to_device(positions, stream=stream)
    d_vdir = cuda.to_device(vdir, stream=stream)
    
    test_kernel[8, 128, stream](d_positions, d_vdir, step_l)
    stream.synchronize()
    positions = d_positions.copy_to_host(stream=stream)
    print("Finished calculating initial positions")
    simulations._write_traj(traj_file, "a", positions)

def test_number_of_steps():
    mesh = meshio.read("/content/drive/MyDrive/mresprojectbits/vascular_mesh_22-10-04_21-52-57_r4.ply")
    vertices = mesh.points.astype(np.float32)*1e-6
    faces = mesh.cells[0].data

    #load velocity directions and shift to origin
    vdir = np.load("/content/drive/MyDrive/mresprojectbits/velocity_direction.npy")
    vloc = np.load("/content/drive/MyDrive/mresprojectbits/velocity_location.npy")
    vloc = vloc*1e-6
    padding=np.zeros(3)
    shift = -np.min(vertices, axis=0) + padding
    vdir = vdir + shift
    vloc = vloc + shift 
    mag = np.linalg.norm(vdir, axis=1)
    mag = mag[:, np.newaxis]
    vdir = np.divide(vdir, mag)

    substrate = substrates.mesh(
        vertices,
        faces,
        periodic=True)

    n_walkers = 100
    traj_file = "IVIM_tests.txt"
    flow_velocity = 0.0030
    gradient = np.zeros((1, 100, 3))
    gradient[0, 1:30, 0] = 1
    gradient[0, 70:99, 0] = -1
    T = 80e-3 
    n_t = int(1e2)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)

    bs = np.linspace(0, 3e9, 50)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient = gradients.set_b(gradient, dt, bs)

    #Calculate flow velocity 
    flow_length = dt*flow_velocity

    signals = simulations.flow_simulation(
        n_walkers, 
        flow_length, 
        gradient, 
        dt, 
        substrate, 
        traj_file, 
        vdir, 
        vloc, 
    )
    trajectories = np.loadtxt(traj_file)
    print(trajectories.shape)
    return 