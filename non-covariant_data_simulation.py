'''This code provides functions to simulate non-covariant ion distributions,
both isotropic and anisotropic. The functions used to run the simulation are
isotropic_simulation and anisotropic simulation. Both start with a list of
variable parameters which can be modified as required.

At present, both datasets are generated using a Monte Carlo simulation. For larger
simulations, this doesn't scale particularly well, so another option is to simulate
an image using the functional form of the image, and point pick using the probability
distribution of the image.
'''

from numba import jit
import numpy as np
import math
import os
from tqdm import tqdm

####################
# Example use cases

def isotropic_simulation():
    '''Note this uses a Monte Carlo simulation'''
    #############################################
    # SET PARAMETERS
    #############################################
    # These are the basic parameters for an isotropic simulation, in format [ion_1, ion_2,...,ion_n]
    # Radius (in 3D) at which ions will be seen, and standard deviation
    radius = [80,30]
    radius_sigma = [4, 1]
    # Mean number of ions per cycle (lambda for a Poisson distribution)
    mean_points = [3, 2.5]
    # Probability the ion is detected
    detection_efficiency = [1., 0.5]
    # Arrival time (PImMS timebin, from 0-4095)
    arrival_time = [1250, 1500]
    # Number of experimental cycles to simulate
    no_cycles = 20000

    filename = 'isotropic_test.bin'

    #############################################
    # Setup
    filename = os.path.join(os.path.dirname(__file__), 'Simulated data', filename)
    rng_helper = np.random.default_rng()
    # Make clean file
    with open(filename, 'wb') as f: pass

    data_array = []

    for _cycle in tqdm(range(no_cycles)):
        # Pick number of events in the cycle from a Poisson distribution.
        # This is for each ion, since not correlated.
        no_events = [rng_helper.poisson(_) for _ in mean_points]

        # Apply detection efficiency before initialising data array and generating data for each event
        no_events = [np.sum(rng_helper.random(_) <= detection_efficiency[i]) for i, _ in enumerate(no_events)]

        # Initialise data array in the correct format for saving
        data = np.zeros((np.sum(no_events), 3), dtype='int')

        # Assign arrival times (these can have a blur applied if required)
        data[:,2] = np.repeat(arrival_time, no_events)

        # Populate the data array
        for i in range(len(no_events)):
            if no_events[i] == 0: continue
            data_end = int(np.sum(no_events[:i+1]))
            data_start = data_end - no_events[i]
            data[data_start:data_end,:2] = np.array([generate_non_covariant_coordinates_isotropic(radius[i], radius_sigma[i]) for j in range(data_end - data_start)])

        data_array.append(data)

        if _cycle % 1000 == 0: 
            save_data(filename, data_array)
            data_array = []

    save_data(filename, data_array)


def anisotropic_simulation():
    '''Note this function uses a Monte Carlo simulation.
    
    The intensity (as a function of theta) is described using an expansion of Legendre polynomials,
    i.e. I(theta) = b_0 L_0(cos(theta)) + b_1 L_1(cos(theta)) + b_2 L_2(cos(theta)) + ...
    where b_n is the n^th beta parameter and L_n is the n^th Legendre polynomial. The first term
    does not vary as a function of theta, since L_0(x) = 1.

    In our experiments, the first three terms of the expansion are usually sufficient to describe the
    distribution. The isotropic case can be recovered by setting b_1, b_2, ... = 0.
    '''
    #############################################
    # SET PARAMETERS
    #############################################
    # These are the basic parameters for an anisotropic simulation, in format [ion_1, ion_2,...,ion_n]
    # Radius (in 3D) at which ions will be seen, and standard deviation
    radius = [80,30]
    radius_sigma = [4, 1]
    # Mean number of ions per cycle (lambda for a Poisson distribution)
    mean_points = [3, 2.5]
    # Anisotropy parameter, list of [b_0, b_1, b_2, ...]. See comment above for more description of this
    beta_params = [[1, -1], [1,0,2]]
    # Probability the ion is detected
    detection_efficiency = [1., 0.5]
    # Arrival time (PImMS timebin, from 0-4095)
    arrival_time = [1250, 1500]
    # Number of experimental cycles to simulate
    no_cycles = 20000

    filename = 'anisotropic_test.bin'

    #############################################
    # Setup
    filename = os.path.join(os.path.dirname(__file__), 'Simulated data', filename)
    rng_helper = np.random.default_rng()
    # Make clean file
    with open(filename, 'wb') as f: pass

    # Get cumulative probability distributions
    cumulative_prob_dists = [get_cumulative_probability_distribution(_) for _ in beta_params]

    data_array = []

    for _cycle in tqdm(range(no_cycles)):
        # Pick number of events in the cycle from a Poisson distribution.
        # This is for each ion, since not correlated.
        no_events = [rng_helper.poisson(_) for _ in mean_points]

        # Apply detection efficiency before initialising data array and generating data for each event
        no_events = [np.sum(rng_helper.random(_) <= detection_efficiency[i]) for i, _ in enumerate(no_events)]

        # Initialise data array in the correct format for saving
        data = np.zeros((np.sum(no_events), 3), dtype='int')

        # Assign arrival times (these can have a blur applied if required)
        data[:,2] = np.repeat(arrival_time, no_events)

        # Populate the data array
        for i in range(len(no_events)):
            if no_events[i] == 0: continue
            data_end = int(np.sum(no_events[:i+1]))
            data_start = data_end - no_events[i]
            data[data_start:data_end,:2] = np.array([generate_non_covariant_coordinates_anisotropic(radius[i], radius_sigma[i], cumulative_prob_dists[i]) for j in range(data_end - data_start)])

        data_array.append(data)

        if _cycle % 1000 == 0: 
            save_data(filename, data_array)
            data_array = []

    save_data(filename, data_array)


##################################
# Isotropic section (Monte Carlo)

@jit(nopython=True, cache=True)
def generate_non_covariant_coordinates_isotropic(radius, r_sigma, img_centre=161.5):
    '''Pick points on an isotropic sphere

    Requires the radius at which the signal should occur, the associated standard deviation,
    and the centre of the image (assuming square).
    '''
    # Pick points on a sphere of radius one.
    # This is done by picking randomly from three normal distributions and normalising.
    coords = np.random.randn(3)
    coords /= np.sum(coords ** 2) ** 0.5

    # Scale the normalised coords to the correct radius and apply sigma.
    # Remember sigma is applied to r, so needs splitting into components
    coords = coords * np.random.normal(radius, r_sigma)

    x = int(np.round(coords[0] + 161.5))
    y = int(np.round(coords[1] + 161.5))
    
    return x, y


####################################
# Anisotropic section (Monte Carlo)

@jit(nopython=True, cache=True)
def generate_non_covariant_coordinates_anisotropic(radius, r_sigma, cumulative_prob_dist, img_centre=161.5):
    '''Pick points on an anisotropic sphere

    Requires the radius at which the signal should occur, the associated standard deviation,
    the cumulative probability distribution associated beta parameters, and the centre of the
    image (assuming square).
    '''
    coords = np.zeros(3)

    # Pick a point from the cumulative probability distribution. This defines the z coordinate
    index = np.abs(cumulative_prob_dist - np.random.uniform(0., 1.)).argmin()

    # Picked from 10000 z spaced between -1 and 1. Convert index to a z coodinate
    coords[2] = index / 10000 * 2 - 1 

    # Equations are:
    # z = r cos(phi), x = r sin(phi) cos(theta), y = r sin(phi) sin(theta)
    # r^2 = x^2 + y^2 + z^2, let r = 1 (normalised)
    # Pick a random theta, since there is an axis of cylindrical symmetry
    theta = np.random.rand() * np.pi * 2
    xy_radius = (1 - coords[2] ** 2) ** 0.5
    # Get the x and y coordinates
    coords[:2] = xy_radius * np.array((np.cos(theta), np.sin(theta)))

    # Scale the normalised coords to the correct radius and apply sigma.
    # Remember sigma is applied to r, so needs splitting into components
    coords = coords * np.random.normal(radius, r_sigma)

    # Make sure the axis of cylindrical symmetry is in the returned plane
    x = int(np.round(coords[0] + 161.5))
    z = int(np.round(coords[2] + 161.5))
    
    return z, x

@jit(nopython=True, cache=True)
def generate_cumulative_probability_distribution(distribution):
    '''Convert probability distribution into a cumulative probabiliry distribution'''
    # Get the cumulative probability distribution as a function of cos(phi)
    # Remove negative values
    distribution[distribution < 0] = 0
    
    # Convert into a probability
    probability = distribution / np.sum(distribution)

    # Generate the cumulative probability function
    cumulative_probability = np.array([np.sum(probability[:i+1]) for i in range(len(probability))])

    # Need the cos of phi to generate the z coordinate
    return cumulative_probability


def leg(l, x):
    #Calculate the Legendre polynomial l (only works up to 28, after that the
    #precision causes issues)
    #Uses 1/(2^n*n!) * d^n/dx^n (x^2 - 1)^n to generate the polynomials
    legendre_poly = np.poly1d([1,0,-1]) ** l

    for i in range(l):
        legendre_poly = legendre_poly.deriv() / 2
    return np.divide(legendre_poly(x), math.factorial(l))

def get_cumulative_probability_distribution(beta_array):
    '''Determine the cumulative probability distribution for a given beta array

    Requires an array of beta in a list or tuple
    '''
    # Generate cos(phi). Note that this is not uniform phi, but is good enough.
    cos_phi = 2 * np.linspace(0, 1, 10000) - 1

    # Build anisotropic distribution
    for n, beta_n in enumerate(beta_array):
        try:
            distribution
            distribution += beta_n * leg(n, cos_phi)
        except NameError:
            distribution = beta_n * leg(n, cos_phi)

    # Get the cumulative probability distribution
    return generate_cumulative_probability_distribution(distribution)


###############
# Saving data
def save_data(filename, data_array):
    '''Saves data from an acquisition cycle

    Filename is the name of the file to append to, and data is the x,y,t array.
    data should be a numpy array with format ((x_0, y_0, t_0), (x_1, y_1, t_1), ...)
    '''
    with open(filename, 'ab') as f:
        for data in data_array:
            np.array([len(data),3]).astype('int32').tofile(f)
            data.astype('uint16').tofile(f)


if __name__ == '__main__':
    isotropic_simulation()
    anisotropic_simulation()