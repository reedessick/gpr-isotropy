__doc__ = "a module to house basic utilities for the library"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np
import healpy as hp

#-------------------------------------------------

TWOPI = 2*np.pi
LOG2PI = np.log(TWOPI)

DEFAULT_NSIDE = 4 ### corresponds to 192 pixels

DEFAULT_NUM_SAMPLES = 10000
DEFAULT_NUM_WALKERS = 50
DEFAULT_NUM_THREADS = 1

DEFAULT_OUTPUT_PATH = 'gpr_isotropy-posterior.hdf5'

#-------------------------------------------------

def read_map(path, nside=None, coord=None, verbose=False):
    """
    read in the map and return it
    if nside is supplied, resample the map to match
    if coord is supplied, check that the map is in the correct coordinate system
    """
    if verbose:
        print('reading: '+path)
    m, h = hp.read_map(path, h=True, verbose=verbose)
    h = dict(h)
    if (coord is not None) and h.has_key('COORD') and (h['COORD'] != coord):
        raise ValueError('%s has COORD=%s, but we require COORD=%s'%(path, h['COORD'], coord))
    if nside is not None:
        if verbose:
            print('    resampling map')
        m = hp.ud_grade(m, nside, power=-2) ### preserve the map's sum
    return m
