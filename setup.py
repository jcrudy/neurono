from distutils.core import setup
from distutils.extension import Extension
import numpy
import sys

#Determine whether to use Cython
if '--cythonize' in sys.argv:
    cythonize_switch = True
    del sys.argv[sys.argv.index('--cythonize')]
else:
    cythonize_switch = False

#Find all includes
local_inc = 'pyearth'
numpy_inc = numpy.get_include()

#Set up the ext_modules for Cython or not, depending
if cythonize_switch:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension("neurono._neurono", ["neurono/_neurono.pyx"],include_dirs = [numpy_inc]),
                             ])
else:
    ext_modules = [Extension("neurono._neurono", ["neurono/_neurono.c"],include_dirs = [numpy_inc]),
                   ]
    
#Create a dictionary of arguments for setup
setup_args = {'name':'neurono',
    'version':'0.1.0',
    'author':'Jason Rudy',
    'author_email':'jcrudy@gmail.com',
    'packages':['neurono','neurono.test'],
    'license':'LICENSE.txt',
    'description':'Some simple neural networks.',
    'long_description':open('README.md','r').read(),
    'py_modules' : ['neurono.neurono'],
    'ext_modules' : ext_modules,
    'classifiers' : ['Development Status :: 3 - Alpha'],
    'requires':['numpy']} 

#Add the build_ext command only if cythonizing
if cythonize_switch:
    setup_args['cmdclass'] = {'build_ext': build_ext}

#Finally
setup(**setup_args)
