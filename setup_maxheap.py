from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
setup(
    ext_modules = \
    cythonize("maxheap.pyx",
              annotate=True,
              compiler_directives={
                  "language_level": 3,
                  "boundscheck": False,
                  "wraparound": False,
                  "initializedcheck": False,
              }
             ),
    include_dirs=[np.get_include()]
)
