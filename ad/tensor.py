from ad.op import Op


class Tensor:
    def __init__(self, val, name=''):
        self.name = name
        self.op = Op('')
        self.value = val
        self.inputs = []
        self.outputs = []
        self.grad = 0
        self.require_grad = False

    def backward(self):
        if self.require_grad:
            self.op.backward(self.grad, self.inputs)
            for input_tensor in self.inputs:
                input_tensor.backward()
