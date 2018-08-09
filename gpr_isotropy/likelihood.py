__doc__ = "a module housing the likelihood functions used in our inference of anisotropies"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import healpy as hp
import numpy as np

#-------------------------------------------------

class Likelihood(object):
    """
    general class representing a likelihood
    is instantiated with references to data to avoid re-computing things
    """

    def __init__(self, maps, exposure):
        ### set up the derived products we need from the maps
        self._npix = len(maps[0])
        self._nside = hp.npix2nside(self._npix)
        self._sum = np.zeros(self._npix, dtype=float)
        self._out = np.zeros((self._npix, self._npix), dtype=float)
        for m in maps:
            assert len(m)==self._npix, 'inconsistent shapes for maps!'
            self._sum += m
            self._out += np.outer(m, m)
       
        assert len(exposure)==self._npix, 'inconsistent shape for exposure!'
        self._exposure = exposure

    def __call__(self, *args):
        return 0. ### flat likelihood

class Ro_Eps(Likelihood):
    """
    likelihood for (Ro, eps) given maps, exposure
    """
    def __call__(self, Ro, eps): 
        raise NotImplementedError

class Ro(Likelihood):
    """
    log p(data|Ro; maps, exposure, kernel)

    analytic marginalization over eps
    """
    def __cal__(self, Ro, kernel):
        raise NotImplementedError

def Eps(Likelihood):
    """
    log p(data|eps; maps, exposure, kernal)

    importance sampling to marginalize over Ro
    if Ro_samples is not supplied, we'll sample some from loglike_Ro
    """
    def __init__(self, maps, exposure):
        self._likelihood_Ro_eps = Ro_Eps(maps, exposure)
        self._npix = self._likelihood_Ro_eps._npix
        self._nside = self._likelihood_Ro_eps._nside
        self._sum = self._likelihood_Ro_eps._sum
        self._out = self._likelihood_Ro_eps._out
        self._exposure = self._likelihood_Ro_eps._exposure

    def __call__(self, eps, Ro_samples):
        ans = np.ans([self._likelihood_Ro_eps(Ro, eps) for Ro in Ro_samples])
        m = np.max(ans)
        return np.log(np.sum(np.exp(ans-m))) + m - np.log(len(Ro_samples))
