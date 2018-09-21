__doc__ = "a module to house basic utilities (mostly I/O related) for the library"
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

#-------------------------------------------------

def _params2array(params):
    data = []
    dtype = []
    for key, val in params.items():
        data.append(val)
        dtype.append( (key, type(val)) )
    return np.array(data, dtype=dtype)

def init_hdf5(h5py_file, rmodel, rprior, kernel):
    """
    set up meta-data about the models
    """
    close = isinstance(h5py_file, str) ### whehter or not to close the file after writing
    if close:
        h5py_file = h5py.File(h5py_file, 'w')

    try:
        # rmodel
        rmodel_group = h5py_file.create_group('RoModel')
        rmodel_group.attrs.create('name', rmodel.__name__)
        rmodel_group.attrs.create('ndim', rmode.ndim)
        rmode_params = rmodel_group.create_dataset('params', data=_params2array(rmodel.params))

        # rprior
        rprior_group = h5py_file.create_group('RoPrior')
        rprior_group.attrs.create('name', rprior.__name__)
        rprior_group.attrs.create('nside', rprior.nside, dtype=int)
        rprior_params = rprior_group.create_dataset('params', data=_params2array(rprior.params))

        # kernel
        kernel_group = h5py_file.create_group('Kernel')
        kernel_group.attrs.create('name', kernel.__name__)
        kernel_group.attrs.create('nside', kernel.nside, dtype=int)
        kernel_params = kernel_group.create_dataset('params', data_params2array(kernel.params))
        kernel_cov = kernel_group.create('cov', data=kernel.cov, shape=(kernel.npix, kernel.npix), dtype=float)

    finally:
        if close:
            h5py_file.close()

def into_hdf5(
        h5py_file,
        count,
        pos,
        lnprob,
        rstate,
        rmodel,
        ro_eps,
    ):
    """
    create a new group in the hdf5 file corresponding to this sample
    """
    close = isinstance(h5py_file, str)
    if close:
        h5py_file = h5py.File(h5py_file, 'a')

    try:
        nwakers, ndim = pos.shape

        group = h5py_file.create_group('samples/%d'%count)

        # store attributes about this sample
        group.attrs.create('rstate', rstate)
        group.attrs.create('nwalkers', nwalkers, dtype=int)
        group.attrs.create('ndim', srmodel.ndim, dtype=int)

        # store probability
        pdata = group.create_dataset('lnprob', data=lnprob, shape=(nwalkers,), dtype=float)

        # store position data
        sdataset = group.create_dataset('state', data=pos, shape=(nwalkers, ndim), dtype=float)
    
        # store eps|ro posterior
        npix = rmodel.npix
        means = group.create_dataset('means', shape=(nwalkers, npix), dtype=float)
        covar = group.create_dataset('covar', shape=(nwalkers, npix, npix), dtype=float)
        for w in xrange(nwalkers):
            r = rmodel(pos[w])
            means[w,:] = ro_eps.eps_mean(r)
            covar[w,...] = ro_eps.eps_cov(r)

    finally:
        if close:
            h5py_file.close()

def from_hdf5(h5py_file, count=-1):
    """
    return count, state, lnprob, rstate
    """
    close = isinstance(h5py_file, str) ### need to close the file at the end of reading
    if close:
        h5py_file = h5py.File(h5py_file, 'r')

    ### read what we want
    try:
        group = h5py_file['samples']
        if count == -1:
            count = max([int(i) for i in h5py_file['samples'].keys()])
        group = h5py_file['samples/%d'%count]
        lnprob = group['lnprob'][...]
        state = group['state'][...]
        rstate = group.attrs['rstate']

    finally:
        if close:
            h5py_file.close()

    return count, state, lnprob, rstate
