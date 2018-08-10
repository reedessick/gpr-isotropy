__doc__ = "a module that houses sampling logic for gpr_isotropy"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import h5py
import emcee

### non-standard libraries
from gpr_isotropy import prior
from gpr_isotropy import likelihood
from gpr_isotropy import posterior

#-------------------------------------------------

DEFAULT_NSIDE = 4 ### corresponds to 192 pixels
DEFAULT_NUM_SAMPLES = 10000
DEFAULT_NUM_WALKERS = 50

#-------------------------------------------------

def sample(*args, **kwargs):
    """
    our wrapper around emcee
    handles I/O, etc under-the-hood
    """
    raise NotImplementedError
