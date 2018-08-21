__doc__ = "a module that houses config-parsing logic. Essentially, constructs objects based on specs in an INI"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

from ConfigParser import ConfigParser

#-------------------------------------------------

DEFAULT_NSIDE = 4

#-------------------------------------------------

def parse(path, nside=DEFAULT_NSIDE, verbose=False):
    """
    parse out the prior, likelihood, and posterior parameters from path
    return prior, likelihood, posterior
    """
    if verbose:
        print('reading config: '+path)
    config = ConfigParser()
    config.read(path)

    raise NotImplementedError
