from __future__ import absolute_import



from setuptools import setup, find_packages
from codecs import open


with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pypsa',
    version='0.17.0',
    author='Tom Brown (FIAS, KIT), Jonas Hoersch (FIAS, KIT), Fabian Hofmann (FIAS), Fabian Neumann (KIT), David Schlachtberger (FIAS)',
    author_email='tom.brown@kit.edu',
    description='Python for Power Systems Analysis',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/PyPSA/PyPSA',
    license='GPLv3',
    packages=find_packages(exclude=['doc', 'test']),
    include_package_data=True,
    install_requires=[
        'six >= 1.13.0',
        'numpy',
        'scipy',
        'pandas>=0.24.0',
        'tables',
        'pyomo>=5.3',
        'matplotlib',
        'networkx>=1.10'
    ],
    extras_require = {
        "cartopy": ['cartopy>=0.16'],
        "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme"],
        'gurobipy':['gurobipy']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ])
