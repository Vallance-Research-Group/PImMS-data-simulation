# PImMS-data-simulation
A collection of code used to generate binary PImMS data. This can be used for testing and trialling the PImMS analysis software, or for exploring the effects of changing parameters in a controlled manner. All codes are set up to generate a sample dataset in the Simulated data folder, and have all variable parameters clearly labelled in the script.

## non-covariant_data_simulation.py
This code can be used to simulate PImMS datasets without any correlation between the ions. It contains two simulation functions: isotropic_simulation and anisotropic_simulation. The isotropic simulation generates points uniformly on a sphere, before projecting into two dimensions. The anisotropic simulation generates points on a sphere according to an expansion of Legendre polynomials, before also projecting into two dimensions.

Variable parameters are found within the two functions, and can be adjusted accordingly. Please note that this means the functions must be modified to take the parameters as inputs if you wish to call them from another script.

## covariant_isotropic_simulation.py
This script generates covariant, isotropic data. As an input, a series of r, theta and phi coordinates are provided, and ions are generated at each point before being rotated to a random orientation. This correlates the position of the ions; the correlation will appear when using covariance analysis.

Note that it has been assumed that the dissociation being simulated ocurrs in a single step, so this script cannot be used to simulate two-step processes in its current form.
