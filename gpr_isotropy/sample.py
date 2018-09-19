__doc__ = "a module that houses sampling logic for gpr_isotropy"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import healpy as hp

import h5py
import emcee

### non-standard libraries
from gpr_isotropy.utils import (DEFAULT_NUM_SAMPLES, DEFAULT_NUM_WALKERS, DEFAULT_NUM_THREADS)
from gpr_isotropy import posterior
from gpr_isotropy import likelihood

#-------------------------------------------------

def _into_hdf5(self, h5py_file):
    raise NotImplementedError

def _from_hdf5(self, h5py_file):
    raise NotImplementedError

#------------------------

class Sampler(object):
    """
    a relatively thin wrapper around emcee.PTSampler that adds in some custom capabilities

    samples based on p(Ro|kernel, rprior) and then uses this to compute p(eps|Ro; kernel, rprior)
    writes this into an hdf5 file with a separate dataset for each Ro sample specifying the analytic posterior for eps
    """

    def __init__(
            self,
            nside,
            maps,
            exposure,
            kernel,
            rprior,
            nwalkers=DEFAULT_NUM_WALKERS,
            nthreads=DEFAULT_NUM_THREADS,
        ):

        # model parameters
        self._nside = nside ### set this first so all the other @property.setters can reference it for sanity checks
        self._npix = hp.nside2npix(nside)

        for m in maps:
            assert len(m)==self._npix, 'map length disagrees with npix'
        self._maps = maps

        assert len(exposure)==self._npix, 'exposure length disagrees with npix'
        self._exposure = exposure

        assert kernel.nside==nside, 'kernel.nside disagrees with nside'
        self._kernel = kernel

        assert rprior.nside==nside, 'rprior.nside disagrees with nside'
        self._rprior = rprior

        # make the posterior things we use for reference
        self._ro = posterior.Ro(maps, exposure, kernel, rprior)
        self._ro_eps = posterior.Ro_Eps(maps, exposure, kernel, rprior)

        # sampler parameters
        if nwalkers < 2*self.npix:
            nwalkers = 2*self.npix
            print('WARNING: increasing the number of walkers to 2*npix=%d'%nwalkers)
        self._nwalkers = nwalkers

        ### set up the PTSampler
        self._sampler = emcee.Sampler(
            self.nwalkers,
            self._npix,
            self._ro.__call__, ### Will this work? Is this pickle-able?
            args=[],
            kwargs={},
            threads=num_threads,
        )

        self._count = 0

    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        return self._npix

    @property
    def maps(self):
        return self._maps

    @property
    def exposure(self):
        return self._exposure

    @property
    def kernel(self):
        return self._kernel

    @property
    def rprior(self):
        return self._rprior

    @property
    def Ro(self):
        return self._ro

    @property
    def Ro_Eps(self):
        return self._ro_eps

    @property
    def nwalkers(self):
        return self._nwalkers

    @property
    def sampler(self):
        return self.sampler

    def initialize(self, path=None, verbose=False):
        if path is None:
            raise NotImplementedError('start around the MLE estimates from posterior (based on non-zero eigenvalues)')
        else:
            raise NotImplementedError('read last sample from path and set that as starting point')

    def sample(self, num_samples, verbose=False):
        raise NotImplementedError
