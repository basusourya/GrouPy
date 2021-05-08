
import numpy as np
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.Z2_array import Z2Array

# A transformation in h2z2 can be coded using four integers:
# m in {0, 1}, mirror reflection in the first translation axis or not
# r in {0, 1, 2, 3}, the rotation index
# u, translation along the first spatial axis
# v, translation along the second spatial axis
# We will always store these in the order (m, r, u, v).
# This is called the 'int' parameterization of p4m.

# A matrix representation of this group is given by
# T(u, v) M(m) R(r)
# where
# T = [[ 1, 0, u],
#      [ 0, 1, v],
#      [ 0, 0, 1]]
# M = [[ 1, 0,         0],
#      [ 0, (-1) ** m, 0],
#      [ 0, 0,         1]]
# R = [[ cos(r pi / 2), -sin(r pi /2), 0],
#      [ sin(r pi / 2), cos(r pi / 2), 0],
#      [ 0,             0,             1]]
# This is called the 'hmat' (homogeneous matrix) parameterization of p4m.

# The matrix representation is easier to work with when multiplying and inverting group elements,
# while the integer parameterization is required when indexing gfunc on h2z2.


class H2Z2Array(MatrixGArray):

    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (3,), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'h2z2'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        assert (p == 'int' and data.shape[-1] == 3) or (p == 'hmat' and data.shape[-2:] == (3, 3))

        self._left_actions[H2Z2Array] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_hvec

        super(H2Z2Array, self).__init__(data, p)

    def int2hmat(self, int_data):
        m = int_data[..., 0]
        u = int_data[..., 1]
        v = int_data[..., 2]
        out = np.zeros(int_data.shape[:-1] + (3, 3), dtype=np.int)
        out[..., 0, 0] = (-1) ** m
        out[..., 0, 1] = 0
        out[..., 0, 2] = u
        out[..., 1, 0] = 0
        out[..., 1, 1] = 1
        out[..., 1, 2] = v
        out[..., 2, 2] = 1.
        return out

    def hmat2int(self, hmat_data):
        neg_det_r = hmat_data[..., 1, 0] * hmat_data[..., 0, 1] - hmat_data[..., 0, 0] * hmat_data[..., 1, 1]
        u = hmat_data[..., 0, 2]
        v = hmat_data[..., 1, 2]
        m = (neg_det_r + 1) // 2

        out = np.zeros(hmat_data.shape[:-2] + (3,), dtype=np.int)
        out[..., 0] = m
        out[..., 1] = u
        out[..., 2] = v
        return out


def identity(shape=(), p='int'):
    e = H2Z2Array(np.zeros(shape + (3,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size + (3,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(minu, maxu, size)
    data[..., 2] = np.random.randint(minv, maxv, size)
    return H2Z2Array(data=data, p='int')



def mirror_u(shape=None):
    shape = shape if shape is not None else ()
    mdata = np.zeros(shape + (3,), dtype=np.int)
    mdata[0] = 1
    return H2Z2Array(mdata)



def m_range(start=0, stop=2):
    assert stop > 0
    assert stop <= 2
    assert start >= 0
    assert start < 2
    assert start < stop
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 0] = np.arange(start, stop)
    return H2Z2Array(m)



def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return H2Z2Array(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 3] = np.arange(start, stop, step)
    return H2Z2Array(m)


def meshgrid(m=m_range(), r=r_range(), u=u_range(), v=v_range()):
    m = H2Z2Array(m.data[:, None, None, None, ...], p=m.p)
    r = H2Z2Array(r.data[None, :, None, None, ...], p=r.p)
    u = H2Z2Array(u.data[None, None, :, None, ...], p=u.p)
    v = H2Z2Array(v.data[None, None, None, :, ...], p=v.p)
    return u * v * m * r


# def gmeshgrid(*args):
#    out = identity()
#    for i in range(len(args)):
#        slices = [None if j != i else slice(None) for j in range(len(args))] + [Ellipsis]
#        d = args[i].data[slices]
#        print i, slices, d.shape
#        out *= H2Z2Array(d, p=args[i].p)
#
#    return out
