import numpy as np
from groupy.garray.garray import GArray
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.H2Z2_array import H2Z2Array
from groupy.garray.Z2_array import Z2Array

from groupy.garray.matrix_garray import MatrixGArray


class H2Array(MatrixGArray):

    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (1,), 'mat': (2, 2), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'H2'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[H2Array] = self.__class__.left_action_mat
        self._left_actions[H2Z2Array] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_vec

        super(H2Array, self).__init__(data, p)

    def int2mat(self, int_data):
        m = int_data[..., 0]
        out = np.zeros(int_data.shape[:-1] + self._g_shapes['mat'], dtype=np.int)
        out[..., 0, 0] = (-1) ** m
        out[..., 0, 1] = 0
        out[..., 1, 0] = 0
        out[..., 1, 1] = 1
        return out

    def mat2int(self, mat_data):
        neg_det_r = mat_data[..., 1, 0] * mat_data[..., 0, 1] - mat_data[..., 0, 0] * mat_data[..., 1, 1]
        m = (neg_det_r + 1) // 2

        out = np.zeros(mat_data.shape[:-2] + self._g_shapes['int'], dtype=np.int)
        out[..., 0] = m
        return out


class H2Group(FiniteGroup, H2Array):

    def __init__(self):
        H2Array.__init__(
            self,
            data=np.arange(2)[:, None],
            p='int'
        )
        FiniteGroup.__init__(self, H2Array)

    def factory(self, *args, **kwargs):
        return H2Array(*args, **kwargs)


H2 = H22Group()

# Generators & special elements
m1 = H2Array(data=np.array([0, 1]), p='int')
e = H2Array(data=np.array([0, 0]), p='int')


def identity(shape=(), p='int'):
    e = H2Array(np.zeros(shape + (1,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size + (1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    return H2Array(data=data, p='int')
