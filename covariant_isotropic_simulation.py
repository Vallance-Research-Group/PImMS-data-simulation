'''This code provides functions to simulate isotropic covariance distributions, for
a concerted process. For each atom in the molecule, the recoil angle (theta and phi)
and the radius at the detector, along with a standard deviation in the position, is
defined. Multiple molecules can be defined in the same simulation.

Parameters are set in the __init__ of the covariant isotropic class
'''

import numpy as np
import math
import os
from tqdm import tqdm
from subfunctions.covariant_isotropic_simulation_numba import get_data, rotate_molecule_convert_PImMS

class covariant_isotropic():
    def __init__(self):
        #######################################################################
        # These are the basic parameters for an isotropic simulation
        # Molecule-specific parameters
        # Array is [[radius, theta, phi, sigma, arrival time]]
            # Radius is the radial distance from the centre in pixels (0-160)
            # Theta is the angle from z axis about y (azimuthal angle) (0-\pi)
            # Phi is the angle from x axis about z (polar angle) (0-2\pi)
            # Sigma is the standard deviation of the uncertainty in ion position, in pixels
            # Arrival time is the ion arrival time in PImMS timebins (0-4095)
        #######################################################################
        self.molecules = [
            [
                [50, 0, 0, 4, 1500],
                [80, np.pi, 0, 2, 1600],
            ],
            [
                [60, 0, 0, 4, 1520],
                [100, 2*np.pi/3, 2*np.pi/3, 1, 1620],
            ],
            [
                [120, 0, 0, 3, 1550],
                [70, np.pi/2, 0, 6, 1650],
            ],
            [
                [50, 0, 0, 4, 1700],
                [80, 1.91, 2*np.pi/3, 2, 1750],
                [60, 1.91, 0, 4, 1800],
                [70, 1.91, 4*np.pi/3, 2, 1850],
            ],
            ]

        # Detection efficiency is the probability a given ion is detected
        # Mean molecule count is the mean number of each molecule dissociating in an acquisition cycle.
        #    This can either be a single value (in a list), or a list of values for each molecule 
        # No cycles is the number of experimental cycles to simulate
        # filename is the output filename
        self.detection_efficiency = 0.5
        self.mean_molecule_count = [0.01, 0.1, 5, 1]
        self.no_cycles = 100000
        filename = 'isotropic_covariance_test.bin'

        # Set the file path relative to this script
        self.filename = os.path.join(os.path.dirname(__file__), 'Simulated data', filename)
        #######################################################################

    def __call__(self):
        # Setup
        rng_helper = np.random.default_rng()

        # Check total number of molecules is consistent across all inputs
        no_molecules = len(self.molecules)
        if len(self.mean_molecule_count) != no_molecules:
            if len(self.mean_molecule_count) == 1:
                self.mean_molecule_count = self.mean_molecule_count * no_molecules

            else:
                print('Input Error: Number of molecules not consistent in the input. Please update mean_molecule_count.')
                return

        # Get the total ion count for each molecule by examining the molecule list
        ion_count_per_molecule = np.array([len(molecule) for molecule in self.molecules])

        # Make clean file
        with open(self.filename, 'wb') as f: pass
        
        # Convert the initial geometry from polar to cartesian.
        # r, theta, phi replaced by x,y,z, and each molecule stored in a list as a numpy array
        cartesian_molecules = [self.polar_to_cartesian(_) for _ in self.molecules]

        # Convert sigma to per-axis value
        for i in range(no_molecules): cartesian_molecules[i][:,3] /= 3**0.5

        data_array = []

        for _cycle in tqdm(range(self.no_cycles)):
            # Pick number of events in the cycle from a Poisson distribution.
            no_events = np.array([rng_helper.poisson(_) for _ in self.mean_molecule_count])

            # Simulate the data
            data_array.append(get_data(cartesian_molecules, no_events, ion_count_per_molecule, self.detection_efficiency))

            # Save out every 1000 acquisition cycles
            if _cycle % 1000 == 0:
                self.save_data(data_array)
                data_array = []

        self.save_data(data_array)


    ##################################
    def polar_to_cartesian(self, molecule):
        '''Convert polar coordinates to cartesian coordinates
        '''
        np_molecule = np.array(molecule)

        # Extract r, theta, phi
        r = np_molecule[:,0]
        theta = np_molecule[:,1]
        phi = np_molecule[:,2]

        # Calculate x, y and z
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Set to the molecule
        np_molecule[:,0] = x
        np_molecule[:,1] = y
        np_molecule[:,2] = z

        return np_molecule


    ###############
    # Saving data
    def save_data(self, data_array):
        '''Saves data from an acquisition cycle

        Filename is the name of the file to append to, and data is the x,y,t array.
        data should be a numpy array with format ((x_0, y_0, t_0), (x_1, y_1, t_1), ...)
        '''
        with open(self.filename, 'ab') as f:
            for data in data_array:
                np.array([len(data),3]).astype('int32').tofile(f)
                data.astype('uint16').tofile(f)


if __name__ == '__main__':
    data_simulator = covariant_isotropic()
    data_simulator()