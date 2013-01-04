#!/usr/bin/python2.7
## @package processing
#Created on November 27, 2012
#
#@author: Trevor Kuhlengel


#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
#
#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("cy_funcs", ["cy_funcs.pyx"])]
#)

from distutils.core import setup
from Cython.Build import cythonize

setup(name="Fish",
      version='1.0',
      description='Zebrafish Segmentation Toolkit',
      author='Trevor Kuhlengel',
      author_email='tkuhlengel@gmail.com',
      #url='http://www.python.org/sigs/distutils-sig/',
      py_modules=['processing','main','graphing'],
      ext_modules = cythonize("cyfuncs.pyx"),
     )

