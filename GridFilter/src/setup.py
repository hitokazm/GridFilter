from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("agent", ["agent.pyx"])]

setup(
  name = 'Grid Filter App',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
