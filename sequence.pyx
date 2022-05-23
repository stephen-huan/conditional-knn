# cython: profile=False
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef int DOUBLE_SIZE = sizeof(double)

cdef (Sequence *) new_sequence(int size):
    """ Make a new sequence with the given size. """
    cdef:
        int i
        Sequence *sequence

    sequence = <Sequence *>PyMem_Malloc(sizeof(Sequence))
    sequence.data = <void **>PyMem_Malloc(size*sizeof(void *))
    sequence.size = size
    # initialize to null so freeing uninitialized elements is safe
    for i in range(size):
        sequence.data[i] = NULL
    return sequence

cdef (Sequence *) from_list(list arrays):
    """ Make a new sequence from a list of ndarrays. """
    cdef:
        int i
        long[::1] array
        Sequence *sequence

    sequence = <Sequence *>PyMem_Malloc(sizeof(Sequence))
    sequence.size = len(arrays)
    sequence.data = <void **>PyMem_Malloc(sequence.size*sizeof(void *))
    for i in range(sequence.size):
        array = arrays[i]
        sequence.data[i] = <void *> &array[0]
    return sequence

cdef long[::1] size_list(list arrays):
    """ Return a list of sizes from a list of ndarrays. """
    cdef:
        int i, size
        long *sizes

    size = len(arrays)
    sizes = <long *>PyMem_Malloc(size*sizeof(long))
    for i in range(size):
        sizes[i] = arrays[i].shape[0]
    return <long[:size:1]> sizes

cdef void add_item(Sequence *sequence, int i, int size, int dtype=DOUBLE_SIZE):
    """ Add a list of dtype of length size at index i. """
    if 0 <= i < sequence.size:
        sequence.data[i] = PyMem_Malloc(size*dtype)

cdef void cleanup(Sequence *sequence, long *size_list=NULL):
    """ Free dynamically allocated memory associated with sequence. """
    cdef int i
    # free each element of the sequence
    if size_list == NULL:
        for i in range(sequence.size):
            PyMem_Free(sequence.data[i])
    PyMem_Free(sequence.data)
    PyMem_Free(sequence)
    PyMem_Free(size_list)

