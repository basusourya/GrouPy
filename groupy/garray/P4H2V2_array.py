import numpy as np
from groupy.garray.garray import GArray
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.P4H2V2Z2_array import P4H2V2Z2Array
from groupy.garray.Z2_array import Z2Array

from groupy.garray.matrix_garray import MatrixGArray


class P4H2V2Array(MatrixGArray):

    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (3,), 'mat': (2, 2), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'P4H2V2'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[P4H2V2Array] = self.__class__.left_action_mat
        self._left_actions[P4H2V2Z2Array] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_vec

        super(P4H2V2Array, self).__init__(data, p)

    def int2mat(self, int_data):
        m1 = int_data[..., 0]
        m2 = int_data[..., 1]
        out = np.zeros(int_data.shape[:-1] + self._g_shapes['mat'], dtype=np.int)
        out[..., 0, 0] = np.cos(0.5 * np.pi * r) * (-1) ** m1
        out[..., 0, 1] = -np.sin(0.5 * np.pi * r) * (-1) ** m1
        out[..., 1, 0] = np.sin(0.5 * np.pi * r) * (-1) ** m2
        out[..., 1, 1] = np.cos(0.5 * np.pi * r) * (-1) ** m2
        return out

    def mat2int(self, mat_data):
        x1 = mat_data[..., 1, 0]
        y1 = mat_data[..., 1, 1]
        x2 = mat_data[..., 0, 0]
        y2 = mat_data[..., 0, 1]

        r = ((np.arctan2(2*x1*y1, y1**2 - x1**2) / np.pi) % 4).astype(np.int)
        a = np.pi * r / 2
        neg_det_r1 = x2 * np.cos(a) - y2 * np.sin(a)
        neg_det_r2 = x1 * np.sin(a) + y1 * np.cos(a)
        m1 = (neg_det_r1 + 1) // 2
        m2 = (neg_det_r2 + 1) // 2

        out = np.zeros(mat_data.shape[:-2] + self._g_shapes['int'], dtype=np.int)
        out[..., 0] = m1
        out[..., 1] = m2
        out[..., 2] = r
        return out


class P4H2V2Group(FiniteGroup, P4H2V2Array):

    def __init__(self):
        P4H2V2Array.__init__(
            self,
            data=np.array([[[0, 0], [0, 1], [0, 2], [0, 3]], [[1, 0], [1, 1], [1, 2], [1, 3]], [[2, 0], [2, 1], [2, 2], [2, 3]], [[3, 0], [3, 1], [3, 2], [3, 3]]]),
            p='int'
        )
        FiniteGroup.__init__(self, P4H2V2Array)

    def factory(self, *args, **kwargs):
        return P4H2V2Array(*args, **kwargs)


P4H2V2 = P4H2V2Group()

# Generators & special elements
r = P4H2V2Array(data=np.array([0, 1]), p='int')
m1 = P4H2V2Array(data=np.array([1, 0]), p='int')
m1 = P4H2V2Array(data=np.array([2, 0]), p='int')
e = P4H2V2Array(data=np.array([0, 0]), p='int')


def identity(shape=(), p='int'):
    e = P4H2V2Array(np.zeros(shape + (3,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size + (3,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 2, size)
    data[..., 2] = np.random.randint(0, 4, size)
    return P4H2V2Array(data=data, p='int')
