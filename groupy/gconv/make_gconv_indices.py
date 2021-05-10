
# Code for generating indices used in G-convolutions for various groups G.
# The indices created by these functions are used to rotate and flip filters on the plane or on a group.
# These indices depend only on the filter size, so they are created only once at the beginning of training.

import numpy as np

from groupy.garray.C4_array import C4
from groupy.garray.D4_array import D4
from groupy.garray.p4_array import C4_halfshift
from groupy.garray.P4H2V2_array import P4H2V2
from groupy.garray.P4H2_array import P4H2
from groupy.garray.P4V2_array import P4V2
from groupy.garray.H2V2_array import H2V2
from groupy.garray.H2_array import H2
from groupy.garray.V2_array import V2

from groupy.gfunc.z2func_array import Z2FuncArray
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.p4mfunc_array import P4MFuncArray
from groupy.gfunc.H2V2Z2func_array import H2V2Z2FuncArray
from groupy.gfunc.H2Z2func_array import H2Z2FuncArray
from groupy.gfunc.V2Z2func_array import V2Z2FuncArray
from groupy.gfunc.P4H2V2Z2func_array import P4H2V2Z2FuncArray
from groupy.gfunc.P4H2Z2func_array import P4H2Z2FuncArray
from groupy.gfunc.P4V2Z2func_array import P4V2Z2FuncArray

# c4_z2
def make_c4_z2_indices(ksize):
    x = np.random.randn(1, ksize, ksize) # input channel related
    f = Z2FuncArray(v=x) # input channel related

    if ksize % 2 == 0:
        uv = f.left_translation_indices(C4_halfshift[:, None, None, None]) # left_translation = g*i2g at a high-level, # output channel related
    else:
        uv = f.left_translation_indices(C4[:, None, None, None]) # C4 here is the group
    r = np.zeros(uv.shape[:-1] + (1,))
    ruv = np.c_[r, uv]
    return ruv.astype('int32')

# c4_p4
def make_c4_p4_indices(ksize):
    x = np.random.randn(4, ksize, ksize) # 4 relates to stabilizer size of input channel
    f = P4FuncArray(v=x) # input channel related

    if ksize % 2 == 0:
        li = f.left_translation_indices(C4_halfshift[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    else:
        li = f.left_translation_indices(C4[:, None, None, None])
    return li.astype('int32')


# P4H2_z2
def make_P4H2_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize) # 1 relates to stabilizer size of input channel
    f = Z2FuncArray(v=x) # input channel related
    uv = f.left_translation_indices(P4H2.flatten()[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    mr = np.zeros(uv.shape[:-1] + (1,))
    mruv = np.c_[mr, uv]
    return mruv.astype('int32')

# P4H2_P4H2Z2
def make_P4H2_P4H2Z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(8, ksize, ksize) # input channel related
    f = P4H2Z2FuncArray(v=x) # input channel related
    li = f.left_translation_indices(P4H2.flatten()[:, None, None, None]) # output channel related
    return li.astype('int32')

# P4V2_z2
def make_P4V2_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize) # 1 relates to stabilizer size of input channel
    f = Z2FuncArray(v=x) # input channel related
    uv = f.left_translation_indices(P4V2.flatten()[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    mr = np.zeros(uv.shape[:-1] + (1,))
    mruv = np.c_[mr, uv]
    return mruv.astype('int32')

# P4V2_P4V2Z2
def make_P4V2_P4V2Z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(8, ksize, ksize) # input channel related
    f = P4V2Z2FuncArray(v=x) # input channel related
    li = f.left_translation_indices(P4V2.flatten()[:, None, None, None]) # output channel related
    return li.astype('int32')

# P4H2V2_z2
def make_P4H2V2_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize) # 1 relates to stabilizer size of input channel
    f = Z2FuncArray(v=x) # input channel related
    uv = f.left_translation_indices(P4H2V2.flatten()[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    m1m2r = np.zeros(uv.shape[:-1] + (1,))
    m1m2ruv = np.c_[m1m2r, uv]
    return m1m2ruv.astype('int32')

# P4H2V2_P4H2V2Z2
def make_P4H2V2_P4H2V2Z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(16, ksize, ksize) # input channel related
    f = P4H2V2Z2FuncArray(v=x) # input channel related
    li = f.left_translation_indices(P4H2V2.flatten()[:, None, None, None]) # output channel related
    return li.astype('int32')

# H2V2_z2
def make_H2V2_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize) # 1 relates to stabilizer size of input channel
    f = Z2FuncArray(v=x) # input channel related
    uv = f.left_translation_indices(H2V2.flatten()[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    m1m2 = np.zeros(uv.shape[:-1] + (1,))
    m1m2uv = np.c_[m1m2, uv]
    return m1m2uv.astype('int32')

# H2V2_H2V2Z2
def make_H2V2_H2V2Z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(4, ksize, ksize) # input channel related
    f = H2V2Z2FuncArray(v=x) # input channel related
    li = f.left_translation_indices(H2V2.flatten()[:, None, None, None]) # output channel related
    return li.astype('int32')

# H2_z2
def make_H2_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize) # 1 relates to stabilizer size of input channel
    f = Z2FuncArray(v=x) # input channel related
    uv = f.left_translation_indices(H2[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    m1 = np.zeros(uv.shape[:-1] + (1,))
    m1uv = np.c_[m1, uv]
    return m1uv.astype('int32')

# H2_H2Z2
def make_H2_H2Z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(2, ksize, ksize) # input channel related
    f = H2Z2FuncArray(v=x) # input channel related
    li = f.left_translation_indices(H2[:, None, None, None]) # output channel related
    return li.astype('int32')

# V2_z2
def make_V2_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize) # 1 relates to stabilizer size of input channel
    f = Z2FuncArray(v=x) # input channel related
    uv = f.left_translation_indices(V2[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    m2 = np.zeros(uv.shape[:-1] + (1,))
    m2uv = np.c_[m2, uv]
    return m2uv.astype('int32')

# V2_V2Z2
def make_V2_V2Z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(2, ksize, ksize) # input channel related
    f = V2Z2FuncArray(v=x) # input channel related
    li = f.left_translation_indices(V2[:, None, None, None]) # output channel related
    return li.astype('int32')

# d4_z2
def make_d4_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize) # 1 relates to stabilizer size of input channel
    f = Z2FuncArray(v=x) # input channel related
    uv = f.left_translation_indices(D4.flatten()[:, None, None, None]) # this adds the output channel stabilizer # output channel related
    mr = np.zeros(uv.shape[:-1] + (1,))
    mruv = np.c_[mr, uv]
    return mruv.astype('int32')

# d4_p4
def make_d4_p4m_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(8, ksize, ksize) # input channel related
    f = P4MFuncArray(v=x) # input channel related
    li = f.left_translation_indices(D4.flatten()[:, None, None, None]) # output channel related
    return li.astype('int32')


def flatten_indices(inds):
    """
    The Chainer implementation of G-Conv uses indices into a 5D filter tensor (with an additional axis for the
    transformations H. For the tensorflow implementation it was more convenient to flatten the filter tensor into
    a 3D tensor with shape (output channels, input channels, transformations * width * height).

    This function takes indices in the format required for Chainer and turns them into indices into the flat array
    used by tensorflow.

    :param inds: np.ndarray of shape (output transformations, input transformations, n, n, 3), as output by
    the functions like make_d4_p4m_indices(n).
    :return: np.ndarray of shape (output transformations, input transformations, n, n)
    """
    n = inds.shape[-2]
    nti = inds.shape[1]
    T = inds[..., 0]  # shape (nto, nti, n, n)
    U = inds[..., 1]  # shape (nto, nti, n, n)
    V = inds[..., 2]  # shape (nto, nti, n, n)
    # inds_flat = T * n * n + U * n + V
    inds_flat = U * n * nti + V * nti + T
    return inds_flat