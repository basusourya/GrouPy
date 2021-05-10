
import groupy.garray.H2Z2_array as h2z2a
from groupy.gfunc.gfuncarray import GFuncArray


class H2Z2FuncArray(GFuncArray):

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

        i2g = h2z2a.meshgrid(
            m=h2z2a.m_range(),
            u=h2z2a.u_range(self.umin, self.umax + 1),
            v=h2z2a.v_range(self.vmin, self.vmax + 1)
        )
        print("i2g",i2g)

        super(H2Z2FuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gint = g.reparameterize('int').data.copy()
        gint[..., 1] -= self.umin
        gint[..., 2] -= self.vmin

        return gint
