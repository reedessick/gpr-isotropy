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
        self._exposure = exposure
        self._npix = len(exposure)
        self._nside = hp.npix2nside(self._npix)

        ### set up the derived products we need from the maps
        self._maps = maps
        self._num = len(maps)
        self._sum = -exposure
        self._out = np.zeros((self._npix, self._npix), dtype=float)
        for m in maps:
            assert len(m)==self._npix, 'inconsistent shapes for maps!'
            self._sum += m
            self._out += np.outer(m, m)
       
    def __call__(self, *args):
        return 0. ### flat likelihood

#---

class Ro_Eps(Likelihood):
    """
    likelihood for (Ro, eps) given maps, exposure
    """
    def eps_fisher(self, Ro):
        """
        return the fisher matrix for eps|Ro
        """
        return np.transpose(self._out*Ro)*Ro ### this should do what I want...

    def eps_mean(self, Ro):
        """
        return the mean value of eps|Ro
        """
        return np.sum(np.linalg.inv(self.eps_fisher(Ro))*Ro*self._sum, axis=1)

    def __call__(self, Ro, eps):
        return -np.sum(Ro*self._exposure) + self._num*np.log(np.sum(Ro*self._exposure)) \
            + np.sum(eps*Ro*(-self._sum)) \
            -0.5*np.sum(eps*np.sum(self.eps_fisher(Ro)*eps, axis=1))

class Ro(Likelihood):
    """
    log p(data|Ro; maps, exposure, kernel)

    analytic marginalization over eps
    """
    def __cal__(self, Ro, kernel):
        gamma = kernel.icov + np.outer(Ro, Ro)*self._out
        sign, logdet_gamma = np.slogdet(gamma)
        assert sign>0, 'unphysical covariance matrix!'
        igamma = np.linalg.inv(gamma)

        ### analytic marginalization over eps
        return -np.sum(Ro*self._exposure) + self._num*np.log(np.sum(Ro*self._exposure)) \
            + 0.5*np.sum(Ro*(self._sum)*np.sum(igamma*Ro*self._sum, axis=1)) \
            - 0.5*logdet_gamma - 0.5*kernel.logdet_cov

def Eps(Ro_Eps):
    """
    log p(data|eps; maps, exposure, kernal)

    importance sampling to marginalize over Ro
    if Ro_samples is not supplied, we'll sample some from loglike_Ro
    """
    def __call__(self, eps, Ro_samples):
        ans = np.ans([Ro_Eps.__call__(self, Ro, eps) for Ro in Ro_samples])
        m = np.max(ans)
        return np.log(np.sum(np.exp(ans-m))) + m - np.log(len(Ro_samples))
