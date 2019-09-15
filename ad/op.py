class Op:
    def __init__(self, name):
        self.name = name

    def backward(self, grad, inputs):
        pass


class AddOp(Op):
    def backward(self, grad, inputs):
        for i in inputs:
            if i.require_grad:
                i.grad += grad


class SubOp(Op):
    def backward(self, grad, inputs):
        if inputs[0].require_grad:
            inputs[0].grad += grad
        if inputs[1].require_grad:
            inputs[1].grad += -1 * grad


class MulOp(Op):
    def backward(self, grad, inputs):
        if inputs[0].require_grad:
            inputs[0].grad += inputs[1].value
        if inputs[1].require_grad:
            inputs[1].grad += inputs[0].value


class DivOp(Op):
    def backward(self, grad, inputs):
        if inputs[0].require_grad:
            inputs[0].grad += 1 / inputs[1].value
        if inputs[1].require_grad:
            inputs[1].grad += -1 * inputs[0].value
