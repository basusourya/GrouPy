import numpy as np
from groupy.garray.garray import GArray
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.H2V2Z2_array import H2V2Z2Array
from groupy.garray.Z2_array import Z2Array

from groupy.garray.matrix_garray import MatrixGArray


class H2V2Array(MatrixGArray):

    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (2,), 'mat': (2, 2), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'H2V2'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[H2V2Array] = self.__class__.left_action_mat
        self._left_actions[H2V2Z2Array] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_vec

        super(H2V2Array, self).__init__(data, p)

    def int2mat(self, int_data):
        m1 = int_data[..., 0]
        m2 = int_data[..., 1]
        out = np.zeros(int_data.shape[:-1] + self._g_shapes['mat'], dtype=np.int)
        out[..., 0, 0] = (-1) ** m1
        out[..., 0, 1] = 0
        out[..., 1, 0] = 0
        out[..., 1, 1] = (-1) ** m2
        return out

    def mat2int(self, mat_data):
        neg_det_r1 = mat_data[..., 0, 0]
        neg_det_r2 = mat_data[..., 1, 1]
        m1 = (neg_det_r1 + 1) // 2
        m2 = (neg_det_r2 + 1) // 2

        out = np.zeros(mat_data.shape[:-2] + self._g_shapes['int'], dtype=np.int)
        out[..., 0] = m1
        out[..., 1] = m2
        return out


class H2V2Group(FiniteGroup, H2V2Array):

    def __init__(self):
        H2V2Array.__init__(
            self,
            data=np.arange(2)[:, None],
            p='int'
        )
        FiniteGroup.__init__(self, H2V2Array)

    def factory(self, *args, **kwargs):
        return H2V2Array(*args, **kwargs)


H2V2 = H2V22Group()

# Generators & special elements
m = H2V2Array(data=np.array([0, 1]), p='int')
e = H2V2Array(data=np.array([0, 0]), p='int')


def identity(shape=(), p='int'):
    e = H2V2Array(np.zeros(shape + (1,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size + (1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    return H2V2Array(data=data, p='int')
