# -*- coding: utf-8 -*-
import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class ORU(function_node.FunctionNode):

    """Odd Root Unit."""

    def __init__(self, r=1.0):
        self.r = float(r)
        self.ir = float(1/self.r)
        self.r2 = float(r**2)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        self.retain_inputs((0,))
        y = x[0].copy()
        
        y = numpy.sign(y)*((self.r2*numpy.abs(y) + 1)**(self.ir) - 1)
        
        return y,

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        y = cuda.elementwise(
            'T x, T r, T ir, T r2', 'T y',
            'y = x >= 0 ? pow((r2*x + 1),ir) - 1 : (T)(-pow((r2*(-x) + 1),ir) + 1)',
            'oru_fwd')(
                x[0], self.r, self.ir, self.r2)
        
        return y,

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gy, = grad_outputs
        return ORUGrad(self.r).apply((x,))[0] * gy,


class ORUGrad(function_node.FunctionNode):

    """Odd Root Unit gradient function."""

    def __init__(self, r):
        self.r = r
        self.ir = float(1/self.r)
        self.r2 = float(r**2)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        x, = inputs
        
        gx = self.r*((self.r2)*numpy.abs(x) + 1)**(self.ir - 1)
        
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return gx,

    def forward_gpu(self, inputs):
        x, = inputs
        gx = cuda.elementwise(
            'T x, T r, T ir, T r2', 'T gx',
            'gx = r * pow((r2*abs(x) + 1),(ir - 1))',
            'oru_bwd')(
                x, self.r, self.ir, self.r2)
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return gx,

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gx, = self.get_retained_outputs()
        ggx, = grad_outputs
        return ggx * gx * (x.data < 0),


def oru(x, r=1.0):
    """ Odd Root Unit function.
    
    See: https://arxiv.org/pdf/1804.11237
    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        r (float): Parameter. Default is 1.0.
    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.
    """
    return ORU(r=r).apply((x,))[0]