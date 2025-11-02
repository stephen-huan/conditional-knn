# conditional-knn

Source code for the paper [_Sparse inverse Cholesky
factorization of dense kernel matrices by greedy
conditional selection_](https://arxiv.org/abs/2307.11648).

## Installing

Install dependencies from `environment.yml` with [conda](https://conda.io/)
or [mamba](https://mamba.readthedocs.io/en/latest/index.html):

```shell
conda env create --prefix ./.venv --file environment.yml
```

or from an explicit spec file (platform must match):

```shell
conda create --prefix ./.venv --file linux-64-explicit-spec-list.txt
conda activate ./.venv
pip install build setuptools
```

See [managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for more information.

Activate `conda` environment:

```shell
conda activate ./.venv
```

Build [Cython](https://cython.org/) extensions:

```shell
python setup.py build_ext --inplace
```

### Intel oneMKL with conda

We rely on the Intel
[oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
library to provide fast numerical routines.

Make sure that `numpy` and `scipy` also use the
MKL for BLAS and LAPACK by checking the output of

```shell
python -c "import numpy; numpy.__config__.show()"
```

which should show something like

```
blas_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['.../.venv/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['.../.venv/include']
...
lapack_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['.../.venv/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['.../.venv/include']
...
```

and similarly for

```shell
python -c "import scipy; scipy.__config__.show()"
```

`conda install numpy` from the `defaults` or `anaconda` channel (not
`conda-forge`) should work, but it sometimes doesn't play well with
installing `mkl-devel`. It's easiest just to use the `intel` channel.

## Downloading datasets

We use datasets from the [SuiteSparse Matrix
Collection](https://sparse.tamu.edu/), the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets.php),
[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/),
and the book [_Gaussian Processes for Machine
Learning_](https://gaussianprocess.org/gpml/data/). Download the datasets
with the provided [fish](https://fishshell.com/) script:

```shell
chmod +x get_datasets
./get_datasets
```

### OCO-2 data

#### Downloading the dataset

Navigate to the
[OCO-2](https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_SIF_11r/summary) solar
induced fluorescence (SIF) dataset. Note that the (current) latest version of
the dataset is `11r`, but this might change in the future. If the above link
doesn't work, be sure to directly [search](https://disc.gsfc.nasa.gov/datasets)
for the `OCO2_L2_Lite_SIF` dataset.

Click on the "Online Archive" blue button on right and
then on the 2017 folder. Each file is a different day.

Note that in order to [download files](https://disc.gsfc.nasa.gov/data-access),
an [Earthdata](https://urs.earthdata.nasa.gov/home) account must be created.

#### Post-processing

First install [R](https://www.r-project.org/) and
[NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
using your preferred package manger.

In order to install R packages locally, follow the instructions
[here](https://statistics.berkeley.edu/computing/software/R-packages)
to create the default [`R_LIBS_USER`](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/libPaths).

```shell
mkdir -p ~/R/x86_64-pc-linux-gnu-library/4.2/
```

Be sure to replace `x86_64-pc-linux-gnu` and `4.2` with your
specific platform and R version, respectively. Running the
command `R --version` should show you something like the below.

```
R version 4.2.3 (2023-03-15) -- "Shortstop Beagle"
Copyright (C) 2023 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)
```

Next, start `R` and enter the following
commands into the REPL to install the packages.

```R
> install.packages("renv", repos = "https://cloud.r-project.org")
> renv::restore()
```

The data can now be compiled with

```shell
R --file=compile_fluorescence_data.R
```

The `compile_fluorescence_data.R` script is due to
[Joe Guinness](https://github.com/joeguinness/).

## Running

Files can be run as modules:

```shell
python -m experiments.cholesky
python -m figures.factor
python -m tests.cknn_tests
```

## Citation

```bibtex
@article{huan2025sparse,
  title = {Sparse {{Inverse Cholesky Factorization}} of {{Dense Kernel Matrices}} by {{Greedy Conditional Selection}}},
  author = {Huan, Stephen and Guinness, Joseph and Katzfuss, Matthias and Owhadi, Houman and Sch{\"a}fer, Florian},
  year = {2025},
  month = sep,
  journal = {SIAM/ASA Journal on Uncertainty Quantification},
  volume = {13},
  number = {3},
  pages = {1649--1679},
  publisher = {{Society for Industrial and Applied Mathematics}},
  doi = {10.1137/23M1606253}
}
```
