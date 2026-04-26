"""The file provides direct interfaces for matrix multiplications
for Intel MKL runtime library.

Although often, numpy distributions are linked to MKL for the popular
linear algebra ops, sparse algebra always defaults to some basic C++ code
of scipy. Here we explicitly create interfaces to MKL BLAS level 2 and 3
routines.
"""

import numpy as np
from ctypes import c_double, c_int, c_char, POINTER, byref
from prince_cr.config import mkl


def mkl_matdescr():
    """Generate matdescr (matrix description) array required by BLAS calls."""
    from ctypes import c_char, POINTER, byref
    ntrans = byref(c_char(b'n'))  # Non-trans
    trans = byref(c_char(b't'))  # trans
    npmatd = np.chararray(6)
    npmatd[0] = b'G'  # General
    npmatd[3] = b'C'  # C-ordering
    return trans, ntrans, npmatd.ctypes.data_as(POINTER(c_char))


trans, nontrans, matdescr = mkl_matdescr()


def mkl_csrmat_args(csr_mat):
    """Cast pointers of scipy csr matrix for MKL BLAS."""
    data = csr_mat.data.ctypes.data_as(POINTER(c_double))
    ci = csr_mat.indices.ctypes.data_as(POINTER(c_int))
    pb = csr_mat.indptr[:-1].ctypes.data_as(POINTER(c_int))
    pe = csr_mat.indptr[1:].ctypes.data_as(POINTER(c_int))
    m = byref(c_int(csr_mat.shape[0]))
    n = byref(c_int(csr_mat.shape[1]))
    return m, n, data, ci, pb, pe


def mkl_pointer(numpy_array):
    """Cast pointers of numpy dense array of double for MKL."""
    from ctypes import c_double, POINTER
    return numpy_array, numpy_array.ctypes.data_as(POINTER(c_double))


def csrmm(alpha, mA, mB, beta, mC, transa=False):
    """Sparse matrix X dense matrix multiplication.

    Returns C = alpha*A*B+beta*C (in C's memory)
    """
    global trans, nontrans, matdescr
    alpha = byref(c_double(alpha))
    beta = byref(c_double(beta))

    tr = trans if transa else nontrans
    m, k, data, ci, pb, pe = mkl_csrmat_args(mA)
    n = byref(c_int(mC.shape[1]))
    b = mB.ctypes.data_as(POINTER(c_double))
    c = mC.ctypes.data_as(POINTER(c_double))
    ldb = byref(c_int(mB.shape[1]))
    ldc = byref(c_int(mC.shape[1]))
    mkl.mkl_dcsrmm(
        tr, m, n, k, alpha, matdescr,
        data, ci, pb, pe,
        b, ldb,
        beta, c, ldc)
    return mC


def csrmv(alpha, mA, b, beta, c, transa=False):
    """Sparse matrix X dense vector multiplication.

    Returns c = alpha*A*b+beta*c (in C's memory)
    """
    alpha = byref(c_double(alpha))
    beta = byref(c_double(beta))
    tr = trans if transa else nontrans
    m, n, data, ci, pb, pe = mkl_csrmat_args(mA)
    bp = b.ctypes.data_as(POINTER(c_double))
    cp = c.ctypes.data_as(POINTER(c_double))
    mkl.mkl_dcsrmv(
        tr, m, n, alpha, matdescr,
        data, ci, pb, pe,
        bp, beta, cp)
    return c


def dgemv(alpha, mA, vb, beta, vc, transa=False):
    """Dense matrix X dense vector multiplication.

    Declarations not evident from ctypes:
    CblasRowMajor=101, CblasColMajor=102
    CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113

    Returns c = alpha*A*B+beta*c (in c's memory)
    """
    alpha = c_double(alpha)
    beta = c_double(beta)

    # Analyse A layout
    layout = 101 if mA.flags['C_CONTIGUOUS'] else 102
    tr = 112 if transa else 111
    m = mA.shape[0]
    n = mA.shape[1]
    lda = m if layout == 102 else n
    # Increment for the vector (each n-th entry for ex.)
    inc = 1
    a = mA.ctypes.data_as(POINTER(c_double))
    b = vb.ctypes.data_as(POINTER(c_double))
    c = vc.ctypes.data_as(POINTER(c_double))
    mkl.cblas_dgemv(layout, tr, m, n, alpha, a, lda, b,
        inc, beta, c, inc)
    return vc
