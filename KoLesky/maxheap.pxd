ctypedef unsigned long long HEAP_DATA

cdef double __get_distance(HEAP_DATA value)
cdef int __get_id(HEAP_DATA value)

cdef class Heap:

    cdef HEAP_DATA[::1] l
    cdef int[::1] ids
    cdef int size

    cdef HEAP_DATA __pop(self)
    cdef int __update_key(self, int i, double dist)
    cdef int __decrease_key(self, int i, double dist)
