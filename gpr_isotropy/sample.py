__doc__ = "a module that houses sampling logic for gpr_isotropy"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import sys

import h5py
import emcee

import healpy as hp

### non-standard libraries
from gpr_isotropy.utils import (DEFAULT_NUM_SAMPLES, DEFAULT_NUM_WALKERS, DEFAULT_NUM_THREADS)
from gpr_isotropy import posterior
from gpr_isotropy import likelihood

#-------------------------------------------------

def into_hdf5(h5py_file, count, result, ro, ro_eps, rmodel):
    raise NotImplementedError

def from_hdf5(h5py_file, count=-1):
    raise NotImplementedError

#-------------------------------------------------

class RoModel(object):
    """
    a model that converts between a very low-dimensional Ro model and one compatible with sampling at nside for Eps
    """
    _allowed_params = []

    def __init__(self, nside, **params):
        self._nside = nside
        self.params = params

    @property
    def nside(self):
        return self._nside

    @property
    def paramss(self):
        return self._params

    @paramss.setter
    def params(self, new_params):
        assert sorted(new_params.keys())==sorted(self._allowed_params), 'new parameters do not match allowed_params=%s'%(' '.join(self._allowed_params))
        self._params = new_params

    @property
    def ndim(self):
        raise NotImplementedError('child class should overwrite this')

    def __call__(self, params):
        raise NotImplementedError('child class should overwrite this')

class IsoRoModel(RoModel):
    """
    an Ro model that is just a uniform isotropic distribution
    """
    def __init__(self, nside, **params):
        RoModel.__init__(self, nside, **params)
        self._ones = np.ones(hp.nside2npix(self.nside), dtype=float)

    @property
    def ndim(self):
        return 1

    def __call__(self, r):
        return self._ones*r       

class PixRoModel(RoModel):
    """
    an Ro model that deals with pixels
    """
    _allowed_params = ['ro_nside']

    @property
    def ndim(self):
        return hp.nside2npix(self.params['ro_nside'])

    def __call__(self, ro_pix):
        return hp.ud_grade(ro_pix, self.nside, power=-2) ### convert to the appropriate nside model

class YlmRoModel(RoModel):
    """
    an Ro model that deals with spherical harmonics
    """
    _allowed_params = ['ro_nside']

    @property
    def ndim(self):
        return hp.nside2npix(self.params['ro_nside'])

    def __call__(self, ro_alm):
        raise NotImplementedError('convert alm to pixel basis')

#-------------------------------------------------

def logpost(params, ro_model, ro_post):
    """
    a utility function used within the sampler
    """
    return ro_post(ro_model(params))

class Sampler(object):
    """
    a relatively thin wrapper around emcee.PTSampler that adds in some custom capabilities

    samples based on p(Ro|kernel, rprior) and then uses this to compute p(eps|Ro; kernel, rprior)
    writes this into an hdf5 file with a separate dataset for each Ro sample specifying the analytic posterior for eps
    """
    _PROGRESSBAR_WIDTH = 30

    def __init__(
            self,
            nside,
            maps,
            exposure,
            kernel,
            rprior,
            rmodel,
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

        assert rmodel.nside==nside, 'rmodel.nside disagrees with nside'
        self._rmodel = rmodel

        # make the posterior things we use for reference
        self._ro = posterior.Ro(maps, exposure, kernel, rprior)
        self._ro_eps = posterior.Ro_Eps(maps, exposure, kernel, rprior)

        # sampler parameters
        if nwalkers < 2*self.npix:
            nwalkers = 2*self.npix
            print('WARNING: increasing the number of walkers to 2*npix=%d'%nwalkers)
        self._nwalkers = nwalkers

        ### set up the Sampler
        self._sampler = emcee.Sampler(
            self.nwalkers,
            self.rmodel.ndim,
            logpost,
            args=[self.rmodel, self.ro], 
            threads=num_threads,
        )
        self._state = None
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
    def RoModel(self):
        return self._ro_model

    @property
    def nwalkers(self):
        return self._nwalkers

    @property
    def sampler(self):
        return self.sampler

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        assert len(new_state) == self.rmodel.ndim
        self._state = new_state

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self):
        self._count = count

    def initialize(self, path=None, verbose=False):
        if path is None:
            raise NotImplementedError('start around the MLE estimates from posterior (based on non-zero eigenvalues)')
        else:
            if verbose:
                print('loading initial state from last sample in: '+path)
            with h5py.File(path, 'r') as h5py_file:
                self.state = from_hdf5(h5py_file)

    def sample(self, num_samples, verbose=False, path=None):
        """
        run the sampler
        if verbose: print progress to the screen
        if path: write output incrementally to file
        """
        checkpoint = path is not None

        try:
            if checkpoint:
                if verbose:
                    print('checkpointing to: '+path)
                h5py_file = h5py.File(path, 'w')

            for i, result in enumerate(self.sampler.sample(self.state, iterations=num_samples)):
                if verbose:
                    n = int((self._PROGRESSBAR_WIDTH+1) * float(i) / num_samples)
                    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (self._PROGRESSBAR_WIDTH - n)))
                    sys.stdout.flush()

                if checkpoint:
                    into_hdf5(h5py_file, self._count, result, self._ro, self._ro_eps, self._rmodel)

                self._state = result[0]
                self._count += 1

        finally:
            if checkpoint:
                h5py_file.close()

            if verbose:
                sys.stdout.write("\n")
                sys.stdout.flush()
