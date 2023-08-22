"""This module contains code for creating and manipulating gradient arrays.

Gradient arrays are numpy.ndarray instances with shape (number of measurements,
number of time points, 3). Gradient array elements are floats representing the
gradient magnitude in SI units (T/m).
"""

import numpy as np

import utils
#from . import utils

GAMMA = 267.522e6  # Gyromagnetic ratio of the simulated spins


def interpolate_gradient(gradient, dt, n_t):
    """Interpolate the gradient array to have `n_t` time points.

    Parameters
    --------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in the gradient array.
    n_t : int
        The number of time points in the gradient array after interpolation.

    Returns
    -------
    interp_g : numpy.ndarray
        Interpolated gradient array.
    dt : float
        Duration of a time step in the interpolated gradient array.
    """
    T = dt * (gradient.shape[1] - 1)
    dt = T / (n_t - 1)
    interp_g = np.zeros((gradient.shape[0], n_t, 3))
    for i in range(gradient.shape[0]):
        for j in range(3):
            interp_g[i, :, j] = np.interp(
                np.linspace(0, T, n_t),
                np.linspace(0, T, gradient.shape[1]),
                gradient[i, :, j],
            )
    return interp_g, dt


def calc_q(gradient, dt, SGP):
    """Calculate the q-vector array corresponding to the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in the gradient array.

    Returns
    -------
    q : numpy.ndarray
        q-vector array.
    """
    if SGP:
            q = GAMMA * (np.sqrt(2))*(1/np.pi)*np.concatenate(
        (
            np.zeros((gradient.shape[0], 1, 3)),
            np.cumsum(dt * (gradient[:, 1::, :] + gradient[:, 0:-1, :]) / 2, axis=1),
        ),
        axis=1,
    )
    else:        
        q = GAMMA * np.concatenate(
            (
                np.zeros((gradient.shape[0], 1, 3)),
                np.cumsum(dt * (gradient[:, 1::, :] + gradient[:, 0:-1, :]) / 2, axis=1),
            ),
            axis=1,
        )
    return q


def calc_b(gradient, dt, SGP):
    """Calculate b-values of the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in gradient array.

    Returns
    -------
    b : numpy.ndarray
        b-values.
    """
    q = calc_q(gradient, dt, SGP)
    b = np.trapz(np.linalg.norm(q, axis=2) ** 2, axis=1, dx=dt)
    return b


def set_b(gradient, dt, b, SGP=False):
    """Scale the gradient array magnitude to correspond to given b-values.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in gradient array.
    b : float or numpy.ndarray
        b-value or an array of b-values with length equal to n of measurements.
    SGP : bool
        if Short Gradient Pulse change so that the gradients are correct

    Returns
    -------
    scaled_g : numpy.ndarray
        Scaled gradient array.
    """
    b = np.asarray(b)

    if np.any(np.isclose(calc_b(gradient, dt), 0)):
        raise Exception("b-value can not be changed for measurements with b = 0")
    ratio = b / calc_b(gradient, dt, SGP)
    scaled_g = gradient * np.sqrt(ratio)[:, np.newaxis, np.newaxis]
    return scaled_g



def rotate_gradient(gradient, Rs):
    """Rotate the gradient array of each measurement according to the
    corresponding rotation matrix.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    Rs : numpy.ndarray
        Rotation matrix array with shape (n of measurements, 3, 3).

    Returns
    -------
    g : numpy.ndarray
        Rotated gradient array.
    """
    g = np.zeros(gradient.shape)
    for i, R in enumerate(Rs):
        if not np.isclose(np.linalg.det(R), 1) or not np.all(
            np.isclose(R.T, np.linalg.inv(R))
        ):
            raise ValueError(f"Rs[{i}] ({R}) is not a valid rotation matrix")
        g[i, :, :] = np.matmul(R, gradient[i, :, :].T).T
    return g


def distribute_points(n_points):
    """
    A function to sample n-points on a sphere. 
    """
    points = []
    phi = np.pi * (np.sqrt(5) - 1) 

    for i in range(n_points):
        #get the y value
        y = 1 - (i / float(n_points - 1)) * 2  
        #calculate the radius at y
        r = np.sqrt(1 - y * y)  

        theta = phi * i  

        x = np.cos(theta) * r
        z = np.sin(theta) * r
        #get the points 
        points.append((x, y, z))
    return np.array(points)


def create_gradient(delta, DELTA, bs, n_t, rs):
  """
  Create a gradient array

  Parameters
  ----------
  delta : float
    Diffusion encoding time
  DELTA : float
    Diffusion time
  n_t : int
    The number of time points in the generated gradient array
  bvals : float or numpy.ndarray
    b-value or array of b-values
  rs : points sampled on sphere

  Returns
  -------
  gradient : numpy.ndarray
    Gradient array.
  dt : float
    Duration of a time step in the gradient array.
  delta/dt : 
    number of timesteps for little delta 
  """


  gradient = np.zeros((1, n_t, 3, rs.shape[0]))
  T = delta + DELTA
  dt = T / (gradient.shape[1] - 1)

  for i in range(rs.shape[0]):

    gradient[0, 1:int(delta/dt), 0, i] = rs[i, 0]
    gradient[0, -int(delta/dt):-1, 0, i] = -rs[i, 0]
    gradient[0, 1:int(delta/dt), 1, i] = rs[i, 1]
    gradient[0, -int(delta/dt):-1, 1, i] = -rs[i, 1]
    gradient[0, 1:int(delta/dt), 2, i] = rs[i, 2]
    gradient[0, -int(delta/dt):-1, 2, i] = -rs[i, 2]

  gradient_final = np.zeros((bs.shape[0], gradient.shape[1], 3, rs.shape[0])) #array of zeros for final gradient array
  for i in range(gradient.shape[3]): #number of gradient directions
    gradient_new = gradient[..., i]
    gradient_new = np.concatenate([gradient_new for _ in bs], axis=0)
    gradient_new = set_b(gradient_new, dt, bs)
    gradient_final[..., i] = gradient_new #new gradients with new b-values
  return gradient_final, dt, delta/dt

def create_SGP_gradient(bs, n_t, Delta, rs):
  """
  Create a gradient array

  Parameters
  ----------
  delta : float
    Diffusion encoding time
  DELTA : float
    Diffusion time
  n_t : int
    The number of time points in the generated gradient array
  bvals : float or numpy.ndarray
    b-value or array of b-values
  rs : points sampled on sphere

  Returns
  -------
  gradient : numpy.ndarray
    Gradient array.
  dt : float
    Duration of a time step in the gradient array.
  """

  T = Delta

  dt = T/(n_t-1)
  gradient = np.zeros((1, n_t, 3, rs.shape[0]))
  #T = delta + DELTA
  #dt = T / (gradient.shape[1] - 1)

  for i in range(rs.shape[0]):

    gradient[0, 0, 0, i] = rs[i, 0]
    gradient[0, -1, 0, i] = -rs[i, 0]
    gradient[0, 0, 1, i] = rs[i, 1]
    gradient[0, -1, 1, i] = -rs[i, 1]
    gradient[0, 0, 2, i] = rs[i, 2]
    gradient[0, -1, 2, i] = -rs[i, 2]

  print(bs)
  gradient_final = np.zeros((bs.shape[0], gradient.shape[1], 3, rs.shape[0]))
  for i in range(gradient.shape[3]): #number of gradient directions
    gradient_new = gradient[..., i]
    gradient_new = np.concatenate([gradient_new for _ in bs], axis=0)
    gradient_new = set_b(gradient_new, dt, bs, SGP=True)
    gradient_final[..., i] = gradient_new #new gradients with new q-values
  return gradient_final, dt

def pgse(delta, DELTA, n_t, bvals, bvecs):
    """Generate a pulsed gradient spin echo gradient array.

    Parameters
    ----------
    delta: float
        Diffusion encoding time.
    DELTA : float
        Diffusion time.
    n_t : int
        The number of time points in the generated gradient array. 
    bvals : float or numpy.ndarray
        b-value or an array of b-values.
    bvecs : numpy.ndarray
        b-vector or array of b-vectors.

    Returns
    -------
    gradient : numpy.ndarray
        Gradient array.
    dt : float
        Duration of a time step in the gradient array.   
    """
    gradient = np.zeros((1, int(1e6), 3))
    T = delta + DELTA
    dt = T / (gradient.shape[1] - 1)
    gradient[0, 1 : np.round(delta / dt).astype(int), 0] = 1
    gradient[0, -np.round(delta / dt).astype(int) : -1, 0] = -1
    gradient, dt = interpolate_gradient(gradient, dt, n_t)
    gradient = np.concatenate([gradient for _ in bvals], axis=0)
    gradient = set_b(gradient, dt, bvals)
    Rs = np.zeros((len(bvals), 3, 3))
    for i, bvec in enumerate(bvecs):
        Rs[i] = utils.vec2vec_rotmat(np.array([1.0, 0.0, 0.0]), bvec)
    gradient = rotate_gradient(gradient, Rs)
    return gradient, dt



