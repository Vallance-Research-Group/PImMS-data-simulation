'''Numba accelerated functions used to run covariant_isotropic_simulation.py
'''

from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def get_data(cartesian_molecules, no_events, ion_count, detection_efficiency):

    # Initialise data array
    data = np.zeros((np.sum(no_events * ion_count), 3), dtype='int')

    # Initialise a pointer to keep track of the next available space in data
    data_pointer = 0

    # Populate the data array
    for i in range(len(no_events)):
        if no_events[i] == 0: continue

        # Separate molecule data into cartesian part (u), std. dev. and t
        u = cartesian_molecules[i][:,:3]
        sigma = cartesian_molecules[i][:,3]
        t = cartesian_molecules[i][:,4]
        
        for j in range(no_events[i]):
            # Generate the data
            data[data_pointer:data_pointer + ion_count[i], :] = rotate_molecule_convert_PImMS(u, sigma, t)

            # Increment the pointer
            data_pointer += ion_count[i]

    # Determine which ions are detected and save the filtered data
    ion_detection_mask = np.random.rand(data.shape[0]) <= detection_efficiency

    return data[ion_detection_mask]

        
    
@jit(nopython=True, cache=True)
def rotate_molecule_convert_PImMS(u, sigma, t):
    '''This function applies a random rotation to a molecule.

    First, a quaternion is calculated to generate the random rotation. This is then applied to the vector array.
    A Gaussian blur is then applied, and the final vectors converted to PImMS pixels.

    Takes the following inputs:
        u - an Nx3 np array containing the base molecular geometry
        sigma - an N-dimensional array containing the standard deviation of the uncertainty in position for the Nth ion
        t - an N-dimensional array containing the PImMS arrival time'''

    # Generate quaternion to apply a random rotation
    # Generate three numbers from a uniform distribution
    r1, r2, r3 = np.random.random(3)

    # Calculate a quaternion using these values. (Shomake's algorithm)
    # s is the scalar part of the quaternion, w is the vector part
    s = np.sqrt(1 - r1) * np.sin(2 * np.pi * r2)
    w = np.array([
        np.sqrt(1 - r1) * np.cos(2 * np.pi * r2),  # x
        np.sqrt(r1) * np.sin(2 * np.pi * r3),      # y
        np.sqrt(r1) * np.cos(2 * np.pi * r3)       # z
    ])
    
    # Apply quaternion rotation to vector u.
    # v is the final vector
    # Here, we are using the form below for computational efficiency
    # v = u + 2w x (w x u + sv)
    v = u + 2 * np.cross(w, np.cross(w, u) + s*u)

    # Apply Gaussian blur
    v += np.random.randn(len(v),3) * sigma[:, None]

    # Determine coordinates
    coords = np.round(v + 161.5, 0).astype('int')

    # Replace z with t to make into PImMS data
    coords[:,2] = t

    # Return PImMS data
    return coords