# cython: profile=False
import numpy as np
cimport numpy as np

cdef unsigned long long MASK = 0xffffffff00000000

cdef unsigned long long __heap_value(double dist, int i):
    """ Embed a distance and an integer as a single int64 value. """
    # distances are positive float values
    cdef unsigned long long value = (<unsigned long long*> &dist)[0]
    # zero out lower 32 bits, embed integer id in mantissa
    value &= MASK
    value |= i
    return value

cdef double __get_distance(unsigned long long value):
    """ Get the distance part of a value. """
    cdef unsigned long long data = value & MASK
    return (<double*> &data)[0]

cdef int __get_id(unsigned long long value):
    """ Get the id part of a value. """
    return value & ~MASK

cdef void __swap(unsigned long long[:] l, int[:] ids, int i, int j):
    """ Swap two values in the heap. """
    l[i], l[j] = l[j], l[i]
    # update id : index in array mapping
    ids[__get_id(l[i])], ids[__get_id(l[j])] = i, j

cdef void __siftup(unsigned long long[:] l, int[:] ids, int i):
    """ Bubble a value upwards. """
    # O(log n)
    while (i >> 1) > 0:
        parent = i >> 1
        # heap invariant broken, fix
        if l[i] > l[parent]:
            __swap(l, ids, i, parent)
            i = parent
        # heap invariant mantained
        else:
            break

cdef void __siftdown(unsigned long long[:] l, int[:] ids, int i):
    """ Bubble a value downwards. """
    # O(log n)
    size = l.shape[0] - 1
    while (i << 1) <= size:
        left, right = i << 1, (i << 1) | 1
        # pick larger child
        child = left if right > size or l[left] > l[right] else right
        # heap invariant broken, fix
        if l[child] > l[i]:
            __swap(l, ids, i, child)
            i = child
        # heap invariant mantained
        else:
            break

cdef void __heapify(unsigned long long[:] l, int[:] ids):
    """ Ensure heap property of data. """
    # O(n)
    for i in range((l.shape[0] - 1) >> 1, 0, -1):
        __siftdown(l, ids, i)

def resize(array):
    """ Double the size of the array. """
    return np.append(array, np.zeros(len(array), np.asarray(array).dtype))

cdef class Heap:

    cdef unsigned long long[:] l
    cdef int[:] ids
    cdef int size

    def __init__(self, double[:] dists, long[:] ids):
        """ Construct a new heap from the given data. """
        cdef int i
        self.size = dists.shape[0]
        # dummy value at the front
        self.l = np.zeros(self.size + 1, dtype=np.ulonglong)
        for i in range(self.size):
            self.l[i + 1] = __heap_value(dists[i], ids[i])
        self.ids = np.zeros(self.size, dtype=np.int32)
        for i in range(self.size):
            self.ids[ids[i]] = i + 1
        # enforce heap ordering
        __heapify(self.l, self.ids)

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        return str([(__get_distance(x), __get_id(x))
                    for x in np.asarray(self.l[1: 1 + self.size])])

    def __repr__(self) -> str:
        return f"Heap({str(self)})"

    def __getitem__(self, int i):
        return self.l[i + 1]

    def push(self, double dist, int i) -> None:
        """ Add a value to the heap. """
        self.size += 1
        # hit capacity, resize
        if self.size == self.l.shape[0]:
            self.l, self.ids = resize(self.l), resize(self.ids)
        # add data at the end
        self.l[self.size] = __heap_value(dist, i)
        self.ids[i] = self.size
        # restore heap property
        __siftup(self.l, self.ids, self.size)

    def pop(self) -> tuple:
        """ Remove a value from the heap. """
        cdef unsigned long long value = self.l[1]
        # replace with last leaf
        __swap(self.l, self.ids, 1, self.size)
        self.l[self.size] = 0
        self.size -= 1
        # restore heap property
        __siftdown(self.l, self.ids, 1)
        return __get_distance(value), __get_id(value)

    def update_key(self, int i, double dist) -> None:
        """ Update the value of key. """
        cdef int k = self.ids[i]
        cdef float value = __get_distance(self.l[k])
        # overwrite value
        self.l[k] = __heap_value(dist, i)
        # restore heap property
        if self.l[k] < value:
            __siftdown(self.l, self.ids, k)
        if self.l[k] > value:
            __siftup(self.l, self.ids, k)

    def decrease_key(self, int i, double dist) -> None:
        """ Decrease the value of key. """
        cdef int k = self.ids[i]
        cdef float value = __get_distance(self.l[k])
        # only overwrite value if less
        if dist < value:
            self.l[k] = __heap_value(dist, i)
            # restore heap property
            __siftdown(self.l, self.ids, k)

