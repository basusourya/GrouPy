
import groupy.garray.H2V2Z2_array as h2v2z2a
from groupy.gfunc.gfuncarray import GFuncArray


class H2V2Z2FuncArray(GFuncArray):

    def __init__(self, v, umin=None, umax=None, vmin=None, vmax=None):

        if umin is None or umax is None or vmin is None or vmax is None:
            if not (umin is None and umax is None and vmin is None and vmax is None):
                raise ValueError('Either all or none of umin, umax, vmin, vmax must equal None')

            # If (u, v) ranges are not given, determine them from the shape of v,
            # assuming the grid is centered.
            nu, nv = v.shape[-2:]

            hnu = nu // 2
            hnv = nv // 2

            umin = -hnu
            umax = hnu
            vmin = -hnv
            vmax = hnv

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2g = h2v2z2a.meshgrid(
            m1=h2v2z2a.m1_range(),
            m2=h2v2z2a.m2_range(),
            u=h2v2z2a.u_range(self.umin, self.umax + 1),
            v=h2v2z2a.v_range(self.vmin, self.vmax + 1)
        )

        if v.shape[-3] == 4:
            i2g = i2g.reshape(4, i2g.shape[-2], i2g.shape[-1])
            self.flat_stabilizer = True
        else:
            self.flat_stabilizer = False

        super(H2V2Z2FuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gint = g.reparameterize('int').data.copy()
        gint[..., 2] -= self.umin
        gint[..., 3] -= self.vmin

        if self.flat_stabilizer:
            gint[..., 1] += gint[..., 0] * 2
            gint = gint[..., 1:]

        return gint
