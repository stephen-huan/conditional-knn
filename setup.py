from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "*", ["*.pyx"],
        # to cimport numpy
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=\
    cythonize(extensions,
              annotate=True,
              compiler_directives={
                  "language_level": 3,
                  "boundscheck": False,
                  "wraparound": False,
                  "initializedcheck": False,
                  "cdivision": True,
              },
    ),
)
