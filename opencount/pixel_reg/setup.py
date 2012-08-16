from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("distance_transform", ["distance_transform.pyx"])]

setup(
    name = 'distance transform',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
