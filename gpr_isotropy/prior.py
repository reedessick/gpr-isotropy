__doc__ = "a module that houses the priors for our inferences"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np
import healpy as hp

### non-standard libraries
from gpr_isotropy.utils import (TWOPI, LOG2PI)

#-------------------------------------------------

class RoPrior(object):
    """
    general class representing a prior
    """
    _allowed_params = sorted([])

    def __init__(self, nside, **params):
        self._nside = nside
        self._npix = hp.nside2npix(nside)
        self.params = params ### automatically checks that these are the correct params

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

    def __call__(self, *args):
        return 0. 

#-------------------------------------------------

### priors for Ro

class PosSemiDef(RoPrior):
    """
    require Ro to be positive semi-definite (each element must be >= 0)
    """
    def __call__(self, Ro):
        return 0. if np.all(Ro>=0) else -np.infty

class LogNorm(RoPrior):
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

    _allowed_params = sorted(['mean', 'var'])

    def __init__(self, nside, **params):
        RoPrior.__init__(self, nside, **params)

        mean = self.params['mean']
        var = self.params['var']
        assert len(mean)==self._npix or isinstance(mean, (float, int)), 'bad shape for mean'

        if isinstance(var, (float, int)):
            self._norm = 0.5*N*np.log(twopi) - 0.5*self._npix*np.log(var)
        else:
            assert len(var)==self._npix, 'bad shape for var'
            self._norm = 0.5*N*np.log(twopi) - 0.5*np.sum(np.log(var))

    def __call__(self, Ro):
        return np.sum( - 0.5*(np.log(Ro)-self.params['mean'])/self.params['var'] ) - self._norm

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
        self._icov = np.diag(np.ones(self._npix, dtype=float)) ### we don't use self._compute_icov or self._compute_logdet_cov because this is so easy
        self._logdet_cov = 0.

    def _compute_icov(self):
        self._icov = np.linalg.inv(self._cov) ### could be fragile

    def _compute_logdet_cov(self):
        s, self._logdet_cov = np.slogdet(self._cov)
        assert s>0, 'unphysical covariance matrix! sign of the determinant is not positive'

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
        return -0.5*np.sum(eps*np.sum(self._icov*eps, axis=1)) - 0.5*(self._npix*LOG2PI - self._logdet_cov)

    def __add__(self, other):
        """
        add the covariances and then re-compute internals
        returns a new object
        """
        self._is_safe_to_add(other)

        ### make a new object and return it
        new = Kernel(self.nside)      
        new._cov = self._cov + other._cov
        new._compute_icov()
        new._compute_logdet_cov()

        return new
        
    def __iadd__(self, other):
        """
        add covariances and then re-compute internals
        modifies this object in place
        """
        self._is_safe_to_add(other)

        ### modify things in place
        self._cov += other._cov
        self._compute_icov()
        self._compute_logdet_cov()

        return self

    def _is_safe_to_add(self, other):
        assert self.nside==other.nside, 'can only add kernels with the same nside!'

class WhiteKernel(Kernel):
    """
    a white kernel with specifiable variance
    """
    _allowed_params = ['w']

    def _compute(self):
        w = self.params['w']
        self._cov = np.diag(np.one(self._npix, dtype=float))*w**2
        self._icov = np.diag(np.ones(self._npix, dtype=float))/w**2 ### we don't use self._compute_* here because this is so easy
        self._logdet_cov = self._npix*2*np.log(self.params['w'])

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
        self._compute_icov()
        self._compute_logdet_cov()
