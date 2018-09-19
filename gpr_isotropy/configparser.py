__doc__ = "a module that houses config-parsing logic. Essentially, constructs objects based on specs in an INI"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

from ConfigParser import ConfigParser

### non-standard libraries
from gpr_isotropy import sample
from gpr_isotropy import prior

#-------------------------------------------------

KERNEL_SECTION = 'kernel'
RPRIOR_SECTION = 'rprior'
SAMPLER_SECTION = 'sampler'

#-------------------------------------------------

def readparse(path, verbose=False):
    """
    parse out the prior, likelihood, and posterior parameters from path
    return prior, likelihood, posterior
    """
    if verbose:
        print('reading config: '+path)
    config = ConfigParser()
    config.read(path)

    return config

#------------------------

def config2kernel(config, nside, verbose=False):
    '''
    generates an overall kernel based on the sum of what's specified in the config file
    '''
    items = config.items(KERNEL_SECTION)
    assert len(items), 'could not find any kernel specifications in [%s]'%KERNEL_SECTION

    kernel = prior.WhiteKernel(nside, w=0.) ### will get non-sensical icov, logdet_cov but that's fine
                                            ### we'll overwrite that in the loop.
    for kernel, kwargs in items:
        kernel = kernel + _item2kernel(kernel, nside, verbose=verbose, **eval(kwargs)) ### create a new object each time!
                                                                                       ### eval(kwargs) might be fragile...

    return kernel

def _item2kernel(kernel, nside, verbose=False, **kwargs):
    if verbose:
        print('including %s(nside=%d, **%s)'%(kernel, nside, kwargs))
    if kernel=='WhilteKernel':
        return prior.WhiteKernel(nside, **kwargs)
    elif kernel=='SqrExpKernel':
        return prior.SqrExpKernel(nside, **kwargs)
    else:
        raise ValueError('%s not understood!'%kernel)

#------------------------

def config2rprior(config, nside, verbose=False):
    '''
    extracts the rprior parametes similarly to kernel, but only allows for a single model instead of the sum over several
    '''
    items = config.items(RPRIOR_SECTION)
    assert len(items), ' could not find any rprior specifications in [%s]'%RPRIOR_SECTION
    assert len(items)==1, 'cannot parse more than one rprior specification in [%s]'%RPRIOR_SECTION
    
    rprior, kwargs = items[0]
    kwargs = eval(kwargs) ### could be fragile

    if rprior=='PosSemiDef':
        return prior.PosSemiDef(nside, **kwargs)
    elif rprior=='LogNorm':
        return prior.LogNorm(nside, **kwargs)
    else:
        raise ValueError('%s not understood!'%rprior)

#------------------------

def config2sampler(config, nside, maps, exposure, verbose=False):
    '''
    extracts the relevant settings from the config to build a sample.Sampler object
    '''
    ### instantiate priors
    kernel = config2kernel(config, nside, verbose=verbose)
    rprior = config2rprior(config, nside, verbose=verbose)

    if verbose:
        print('generating Posterior')
    ro = posterior.Ro(maps, exposure, kernel, rprior)
    ro_eps = posterior.Ro_Eps(maps, exposure, kernel, rprior)

    if verbose:
        print('generating Sampler')

    kwargs = {}
    if config.has_option('sampler', 'nwalkers'):
        kwargs['nwalkers'] = config.getint('sampler', 'nwalkers')

    if config.has_option('sampler', 'ntemps'):
        kwargs['ntemps'] = config.get('sampler', 'ntemps')
    elif config.has_option('sampler', 'temp_ladder'):
        kwargs['ntemps'] = [float(_) for _ in config.get('sampler', 'temp_ladder').split()]
    
    return sample.Sampler(nside, maps, exposure, kernel, rprior, **kwargs)
