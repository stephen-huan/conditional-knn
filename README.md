# conditional-knn

A repository to explore the usefulness of the
conditional _k_-nearest neighbor estimator.

## Installing

Install dependencies from `environment.yml` with [conda](https://conda.io/):
```shell
conda env create --prefix ./venv --file environment.yml
```
or from a non-explicit spec file (platform may need to match):
```shell
conda create --prefix ./venv --file linux-64-spec-list.txt
```
or from an explicit spec file (platform must match):
```shell
conda create --prefix ./venv --file linux-64-explicit-spec-list.txt
```
See
[managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for more information.

Activate `conda` environment:
```shell
conda activate ./venv
```
Build [Cython](https://cython.org/) extensions:
```shell
python setup.py build_ext --inplace
```

### Downloading datasets

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

#### OCO-2 data

##### Downloading the dataset

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

##### Post-processing

First install [R](https://www.r-project.org/) and
[NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
using your preferred package manger.
```shell
sudo pacman -S r netcdf
```

In order to install R packages locally, follow the instructions
[here](https://statistics.berkeley.edu/computing/software/R-packages)
to create the default [`R_LIBS_USER`](
https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/libPaths).
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
python -m tests.cknn_tests
```

