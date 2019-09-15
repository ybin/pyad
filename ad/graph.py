from ad.tensor import Tensor
import ad.op as op
import numpy as np


def _tensor(value, name, require_grad=False):
    t = Tensor(value, name=name)
    t.require_grad = require_grad
    return t


def _op(value, operation, inputs, name):
    output = _tensor(value, require_grad=any([i.require_grad for i in inputs]), name=name)
    output.op = operation
    output.inputs = inputs
    for i in inputs:
        i.outputs.append(output)
    return output


def constant(value, name=''):
    return _tensor(np.array(value), require_grad=False, name=name)


def variable(value, name=''):
    return _tensor(np.array(value), require_grad=True, name=name)


def add(a, b, name=''):
    return _op(np.array(a.value + b.value), op.AddOp(name=name + '_AddOp'), [a, b], name)


def sub(a, b, name=''):
    return _op(np.array(a.value - b.value), op.SubOp(name=name + '_SubOp'), [a, b], name)


def mul(a, b, name=''):
    return _op(np.array(a.value * b.value), op.MulOp(name=name + '_MulOp'), [a, b], name)


def div(a, b, name=''):
    return _op(np.array(a.value / b.value), op.DivOp(name=name + '_DivOp'), [a, b], name)
