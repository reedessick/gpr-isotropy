__doc__ = "a module that houses the priors for our inferences"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np
import healpy as hp

#-------------------------------------------------

TWOPI = 2*np.pi

#-------------------------------------------------

class Prior(object):
    """
    general class representing a prior
    """
    def __init__(self, nside):
        self._nside = nside
        self._npix = hp.nside2npix(nside)

    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        return self._npix

    def __call__(self, *args):
        return 0. 

#-------------------------------------------------

### priors for Ro

class PosSemiDef(Prior):
    """
    require Ro to be positive semi-definite (each element must be >= 0)
    """
    def __call__(self, Ro):
        return 0. if np.all(Ro>=0) else -np.infty

class LogNorm(Prior):
    """
    log-normal prior for Ro, each direction separately
    assumes
         len(Ro) = len(mean) = len(stdv)
    but can also handle cases where
        mean is a float, int
    or
        var is a float, int
    in which case it applies the same prior to all directions
    """
    def __init__(self, nside, mean, var):
        Prior.__init__(self, nside)

        assert len(mean)==self._npix or isinstance(mean, (float, int)), 'bad shape for mean'
        self._mean = mean

        if isinstance(var, (float, int)):
            self._var = var
            self._norm = 0.5*N*np.log(twopi) - 0.5*self._npix*np.log(var)
        else:
            assert len(var)==self._npix, 'bad shape for var'
            self._var = var
            self._norm = 0.5*N*np.log(twopi) - 0.5*np.sum(np.log(var))

    def __call__(self, Ro):
        return np.sum( - 0.5*(np.log(Ro)-self._mean)/self._var ) - self._norm

#---

### priors for eps

class Kernel(object):
    """
    a representation of a Gaussian Process kernel between different locations on the sky
    assumes a healpix decomposition on the sky
    this base class implements a white kernal with unit variance. Extensions can do more interesting things
    """
    _allowed_params = sorted([])

    def __init__(self, nside, **params):
        self._nside = nside
        self._npix = hp.nside2npix(nside)
        self._theta, self._phi = hp.pix2ang(nside, np.arange(self._npix))
        self.params = params ### automatically computes stuff, which will define private attributes as well

    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        return self._npix

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        assert sorted(new_params.keys())==sorted(self._allowed_params), 'new parameters do not match allowed_params=%s'%(' '.join(self._allowed_params))
        self._params = new_params
        self._compute()

    def _compute(self):
        self._cov = np.diag(np.ones(self._npix, dtype=float))
        self._icov = np.diag(np.ones(self._npix, dtype=float))
        self._logdet_cov = 0.

    @property
    def cov(self):
        return self._cov

    @property
    def icov(self):
        return self._icov

    @property
    def logdet_cov(self):
        return self._logdet_cov

    @property
    def logdet_icov(self):
        return -self._logdet_cov

    def __call__(self, eps):
        """
        evaluate the inner product of the kernal with some vector
        """
        return np.sum(eps*np.sum(self._icov*eps, axis=1))

class WhiteKernel(Kernel):
    """
    a white kernel with specifiable variance
    """
    _allowed_params = ['s']

    def _compute(self):
        self._cov = np.diag(np.one(self._npix, dtype=float))*self.params['s']**2
        self._icov = np.diag(np.ones(self._npix, dtype=float))/self.params['s']**2
        self._logdet_cov = self._npix*2*np.log(self.params['s'])

class SqrExpKernel(Kernel):
    """
    squared exponential kernel as a function of angular separation
    """
    _allowed_params = ['l', 's']

    def _compute(self):
        ### compute the matrix of angular separations
        self._cov = np.empty((self._npix, self._npix), dtype='float')
        costheta = np.cos(self._theta)
        sintheta = np.sin(self._theta)
        for i in xrange(self._npix): ### could also be accomplished via np.outer?
            self._cov[i,:] = costheta[i]*costheta + sintheta[i]*sintheta*np.cos(self._phi[i]-self._phi)

        ### evaluate covariance
        self._cov = self.params['s']**2 * np.exp(-0.5*self._cov/self.params['l'])**2
        self._icov = np.linalg.inv(self._cov) ### NOTE: could be fragile...
        s, self._logdet_cov = np.slogdet(self._cov)
        assert s>0, 'unphysical covariance matrix! sign of the determinant is not positive'
