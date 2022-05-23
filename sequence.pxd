cdef struct Sequence:
    void **data
    int size

cdef (Sequence *) new_sequence(int size)
cdef (Sequence *) from_list(list arrays)
cdef long[::1] size_list(list arrays)
cdef void add_item(Sequence *sequence, int i, int size, int dtype=*)
cdef void cleanup(Sequence *sequence, long *size_list=*)

