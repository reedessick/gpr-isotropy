#!/usr/bin/env python
__usage__ = "setpy.py command [--options]"
__doc__ = "standard install script"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

from distutils.core import setup

setup(
    name = 'iDQ',
    version = '2.0',
    url = 'https://git.ligo.org/reed.essick/iDQ',
    author = __author__,
    author_email = 'reed.essick@ligo.org',
    description = __doc__,
    liscence = 'MIT License',
    scripts = [
        'bin/sample-gpr_isotropy',
        'bin/investigate-complexity',
    ],
    packages = [
        'gpr_isotropy',
    ],
    data_files = [
    ],
    requires = [
    ],
)
