# conditional-knn

A repository to explore the usefulness of the
conditional _k_-th nearest neighbor estimator.

## Installing

Install dependencies from `environment.yml` with [conda](https://conda.io/):
```bash
conda env create --prefix ./venv --file environment.yml
```
or from a non-explicit spec file (platform may need to match):
```bash
conda create --prefix ./venv --file osx-64-spec-file.txt
```
or from an explicit spec file (platform must match):
```bash
conda create --prefix ./venv --file osx-64-explicit-spec-list.txt
```
See
[managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for more information.

Activate `conda` environment:
```bash
conda activate ./venv
```
Build [Cython](https://cython.org/) extensions:
```bash
python setup.py build_ext --inplace
```
Pre-compiled `*.c` files are also provided.

### Intel MKL with conda

Follow the instructions
[here](https://www.intel.com/content/www/us/en/developer/articles/technical/using-intel-distribution-for-python-with-anaconda.html).
in order to use the Intel MKL libraries, the libraries must have been loaded
_before_ the application uses any functions, or `dlopen()` will error:
```
ImportError: dlopen(ccknn.cpython-39-darwin.so, 2): Symbol not found: _vdLn
  Referenced from: ccknn.cpython-39-darwin.so
  Expected in: flat namespace
```

We rely on numpy to load MKL, so make sure it does:
```bash
DYLD_PRINT_LIBRARIES=1 python -c "import numpy" 2>&1 | grep mkl
dyld: loaded: <...> venv/lib/python3.9/site-packages/mkl/_mklinit.cpython-39-darwin.so
dyld: loaded: <...> venv/lib/libmkl_rt.2.dylib
...
```
(Check to see `libmkl_*.dylib` is being loaded)

Also check the output of:
```bash
python -c "import numpy; numpy.__config__.show()"
blas_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['venv/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['venv/include']
...
```
and similarly for
```bash
python -c "import scipy; scipy.__config__.show()"
```

`conda install numpy` from the `defaults` or `anaconda` channel (not
`conda-forge`) should work, but it sometimes doesn't play well with
installing `mkl-devel`. It's easiest just to use the `intel` channel.

## Running

Files can be run as modules:
```bash
python -m tests.cknn_tests
```

