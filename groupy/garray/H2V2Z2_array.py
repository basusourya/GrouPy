
import numpy as np
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.Z2_array import Z2Array

# A transformation in h2v2z2 can be coded using four integers:
# m1 in {0, 1}, mirror reflection in the first translation axis or not
# m2 in {0, 1}, the rotation index
# u, translation along the first spatial axis
# v, translation along the second spatial axis
# We will always store these in the order (m, r, u, v).
# This is called the 'int' parameterization of v2z2.

# A matrix representation of this group is given by
# T(u, v) M(m) R(r)
# where
# T = [[ 1, 0, u],
#      [ 0, 1, v],
#      [ 0, 0, 1]]
# M = [[ (-1) ** m1, 0,         0],
#      [ 0, (-1) ** m2, 0],
#      [ 0, 0,         1]]
# R = [[ cos(r pi / 2), -sin(r pi /2), 0],
#      [ sin(r pi / 2), cos(r pi / 2), 0],
#      [ 0,             0,             1]]
# This is called the 'hmat' (homogeneous matrix) parameterization of p4m.

# The matrix representation is easier to work with when multiplying and inverting group elements,
# while the integer parameterization is required when indexing gfunc on v2z2.


class H2V2Z2Array(MatrixGArray):

    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (4,), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'h2v2z2'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        assert (p == 'int' and data.shape[-1] == 4) or (p == 'hmat' and data.shape[-2:] == (3, 3))

        self._left_actions[H2V2Z2Array] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_hvec

        super(H2V2Z2Array, self).__init__(data, p)

    def int2hmat(self, int_data):
        m1 = int_data[..., 0]
        m2 = int_data[..., 1]
        u = int_data[..., 2]
        v = int_data[..., 3]
        out = np.zeros(int_data.shape[:-1] + (3, 3), dtype=np.int)
        out[..., 0, 0] = (-1) ** m1
        out[..., 0, 1] = 0
        out[..., 0, 2] = u
        out[..., 1, 0] = 0
        out[..., 1, 1] = (-1) ** m2
        out[..., 1, 2] = v
        out[..., 2, 2] = 1.
        return out

    def hmat2int(self, hmat_data):
        neg_det_r1 = -hmat_data[..., 0, 0]
        neg_det_r2 = -hmat_data[..., 1, 1]
        u = hmat_data[..., 0, 2]
        v = hmat_data[..., 1, 2]
        m1 = (neg_det_r1 + 1) // 2
        m2 = (neg_det_r2 + 1) // 2

        out = np.zeros(hmat_data.shape[:-2] + (4,), dtype=np.int)
        out[..., 0] = m1
        out[..., 1] = m2
        out[..., 2] = u
        out[..., 3] = v
        return out


def identity(shape=(), p='int'):
    e = H2V2Z2Array(np.zeros(shape + (4,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size + (4,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 2, size)
    data[..., 2] = np.random.randint(minu, maxu, size)
    data[..., 3] = np.random.randint(minv, maxv, size)
    return H2V2Z2Array(data=data, p='int')



def mirror_u(shape=None):
    shape = shape if shape is not None else ()
    mdata = np.zeros(shape + (4,), dtype=np.int)
    mdata[0] = 1
    return H2V2Z2Array(mdata)



def m_range(start=0, stop=2):
    assert stop > 0
    assert stop <= 2
    assert start >= 0
    assert start < 2
    assert start < stop
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 0] = np.arange(start, stop)
    return H2V2Z2Array(m)



def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return H2V2Z2Array(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 4] = np.arange(start, stop, step)
    return H2V2Z2Array(m)


def meshgrid(m1=m_range(), m2=m_range(), u=u_range(), v=v_range()):
    m1 = H2V2Z2Array(m1.data[:, None, None, None, ...], p=m1.p)
    m2 = H2V2Z2Array(m2.data[None, :, None, None, ...], p=m2.p)
    u = H2V2Z2Array(u.data[None, None, :, None, ...], p=u.p)
    v = H2V2Z2Array(v.data[None, None, None, :, ...], p=v.p)
    return u * v * m1 * m2


# def gmeshgrid(*args):
#    out = identity()
#    for i in range(len(args)):
#        slices = [None if j != i else slice(None) for j in range(len(args))] + [Ellipsis]
#        d = args[i].data[slices]
#        print i, slices, d.shape
#        out *= H2V2Z2Array(d, p=args[i].p)
#
#    return out
