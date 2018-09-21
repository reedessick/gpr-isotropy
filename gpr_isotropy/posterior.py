__doc__ = "a module housing posterior functions used in our inference of anisotropies"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import healpy as hp
import numpy as np

### non-standard libraries
from gpr_isotropy import likelihood
from gpr_isotropy.utils import DEFAULT_NUM_SAMPLES

#-------------------------------------------------

class Posterior(object):
    """
    general class for a posterior
    extensions may skip certain steps because they can do things analytically
    """
    _likelihood_function = likelihood.Likelihood ### does this mean I can't pickle this?

    def __init__(self, maps, exposure, kernel, rprior):
        self._likelihood = self._likelihood_function(maps, exposure)
        self._kernel = kernel
        self._prior = rprior

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def kernel(self):
        return self._kernel

    @property
    def rprior(self):
        return self._rprior

    def __call__(self, Ro, eps):
        return self._likelihood(Ro, eps) + self._kernel(eps) + self._rprior(Ro)

class Ro_Eps(Posterior):
    _likelihood_function = likelihood.Ro_Eps

    def eps_fisher(self, Ro):
        """
        fisher information matrix for eps|Ro a posteriori (includes effect of kernel)
        """
        return self._likelihood.eps_fisher(Ro) + self._kernel._icov

    def eps_cov(self, Ro):
        """
        covariance matrix for eps|Ro a posteriori (includes effect of kernel)
        """
        return np.linalg.inv(self.eps_fisher(Ro))

    def eps_mean(self, Ro):
        """
        mean for eps|Ro a posteriori (includes effect of kernel)
        """
        igamma = np.linalg.inv(self.eps_fisher(Ro))
        return np.sum(igamma*Ro*self._likelihood._sum, axis=1)

class Ro(Posterior):
    """
    log prob(Ro|maps, exposure, kernel, Rprior)
    """
    _likelihood_function = likelihood.Ro

    def __call__(self, Ro):
        return self.likelihood(Ro, self._kernel) + self._rprior(Ro)

#class Eps(Ro_Eps):
#    """
#    log prob(eps|maps, exposure, kernel, Rprior)
#
#    importance sampling to marginalize over Ro
#    if Ro_samples is not supplied, we'll sample some from loglike_Ro
#    """
#    _likelihood_function = likelihood.Eps
#
#    def __init__(self, maps, exposure, kernel, rprior, num_samples=DEFAULT_NUM_SAMPLES):
#        Posterior.__init__(self, *args)
#        self.sample(num_samples=num_samples)
#
#    def sample(self, num_samples=DEFAULT_NUM_SAMPLES):
#        self._rsamples = []
#        raise NotImplementedError('sample from p(Ro|...) to get Ro samples used in numerical marginaliation in __call__')
#
#    def __call__(self, eps):
#        return self._likelihood(eps, Ro_samples) + self._kernel(eps)
