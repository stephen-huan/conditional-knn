# conditional-knn

A repository to explore the usefulness of the
conditional _k_-th nearest neighbor estimator.

## Installing

Install dependencies from `environment.yml` with [conda](https://conda.io/):
```bash
conda env create --prefix ./venv -f environment.yml
```

Activate `conda` environment
```bash
conda activate ./venv
```

Build [Cython](https://cython.org/) extensions:
```bash
python setup_ccknn.py build_ext --inplace
python setup_maxheap.py build_ext --inplace
```

Pre-compiled `*.c` files are also provided.

## Running

Files can be run as modules:
```bash
python -m tests.cknn_tests
```


