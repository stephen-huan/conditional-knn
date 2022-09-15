# conditional-knn

A repository to explore the usefulness of the
conditional _k_-th nearest neighbor estimator.

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
Pre-compiled `*.c` files are also provided.

### Downloading Datasets

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

## Running

Files can be run as modules:
```shell
python -m tests.cknn_tests
```

