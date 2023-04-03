from math import log, exp


class BinaryFunction:
    '''Simple Function class for creating custom binary operations'''
    def __init__(self, func, deriv1, deriv2):
        # Binary Function
        self.func = func

        # Derivative Functions [d/dx, d/dy]
        self.deriv1 = deriv1
        self.deriv2 = deriv2

    def __call__(self, value, other):
        if isinstance(other, Value):
            out = Value(self.func(value.data, other.data),
                        _prev=(value, other))

            def _backward():
                value.grad += out.grad * self.deriv1(value.data, other.data)
                other.grad += out.grad * self.deriv2(value.data, other.data)
        else:
            # Ignore `other` value if it's a not `Value` (no need gradient)
            out = Value(self.func(value.data, other), _prev=(value,))

            def _backward():
                value.grad += out.grad * self.deriv1(value.data, other)

        out._backward = _backward
        out.grad_fn = f'<{self.__class__.__name__}Backward>'
        return out


class Add(BinaryFunction):
    '''Addition operation'''
    def __init__(self):
        def func(a, b):
            return a + b

        def deriv(a, b):
            return 1

        super().__init__(func, deriv, deriv)


class Mul(BinaryFunction):
    '''Multiplication operation'''
    def __init__(self):
        def func(a, b):
            return a * b

        def deriv1(_, b):
            return b

        def deriv2(a, _):
            return a

        super().__init__(func, deriv1, deriv2)


class Div(BinaryFunction):
    '''Division operation'''
    def __init__(self):
        def func(a, b):
            return a / b

        def deriv1(_, b):
            return 1 / b

        def deriv2(a, b):
            return -a / (b * b)

        super().__init__(func, deriv1, deriv2)


class Pow(BinaryFunction):
    '''Exponentiation operation'''
    def __init__(self):
        def func(a, b):
            return a ** b

        def deriv1(a, b):
            return b * a ** (b - 1)

        def deriv2(a, b):
            return a ** b * log(a)

        super().__init__(func, deriv1, deriv2)


class Function:
    '''Simple Function class for creating custom activation functions'''
    def __init__(self, func, deriv):
        # Binary Function
        self.func = func

        # Derivative of a Function [d/dx]
        self.deriv = deriv

    def __call__(self, input):
        out = Value(self.func(input.data), _prev=(input,))

        def _backward():
            input.grad += out.grad * self.deriv(input.data)
        out._backward = _backward
        out.grad_fn = f'<{self.__class__.__name__}Backward>'
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Exp(Function):
    def __init__(self):
        super().__init__(exp, exp)


class ReLU(Function):
    def __init__(self):
        def func(input):
            return max(input, 0)

        def deriv(input):
            return 1 if input > 0 else 0

        super().__init__(func, deriv)


class Sigmoid(Function):
    def __init__(self):
        def func(input):
            return 1 / (1 + exp(-input))

        def deriv(input):
            return func(input) * (1 - func(input))

        super().__init__(func, deriv)


class Tanh(Function):
    def __init__(self):
        def func(input):
            exp1, exp2 = exp(input), exp(-input)
            return (exp1 - exp2) / (exp1 + exp2)

        def deriv(input):
            exp_sum = exp(input) + exp(-input)
            return 4 / exp_sum ** 2

        super().__init__(func, deriv)


class LeakyReLU(Function):
    def __init__(self, slope=0.01):
        def func(input):
            return max(input, input * slope)

        def deriv(input):
            return 1 if input > 0 else slope

        super().__init__(func, deriv)


# Initialize Binary Functions as build-in
_Add = Add()
_Mul = Mul()
_Div = Div()
_Pow = Pow()

# Initialize Main Activation Functions as build-in
_Exp = Exp()
_ReLU = ReLU()
_Tanh = Tanh()
_Sigmoid = Sigmoid()


class Value:
    '''Simple implementation of Value with autograd'''
    def __init__(self, data, _prev=()):
        self.data = data
        self.grad = 0

        # Backward function for autograd
        self._backward = lambda: None

        # Set of previous Values
        self._prev = set(_prev)

        # Full list of childs
        self._childs = None

        # Name of the gradient function used
        self.grad_fn = None

    def get_childs(self):
        # If self._childs already calculated:
        if self._childs is not None:
            return self._childs

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self._childs = topo[::-1]
        return self._childs

    def backward(self):
        # topological order all of the children in the graph
        topo = self.get_childs()

        # calculate the gradient for all childs
        self.grad = 1
        for v in topo:
            v._backward()

    def zero_grad(self):
        # set all gradients to zero
        topo = self.get_childs()
        for v in topo:
            v.grad = 0

    def update(self, lr=0.01):
        # Update all childs using calculated gradients
        topo = self.get_childs()
        for v in topo[1:]:
            v.data -= lr * v.grad

    def exp(self):
        return _Exp(self)

    def relu(self):
        return _ReLU(self)

    def tanh(self):
        return _Tanh(self)

    def sigmoid(self):
        return _Sigmoid(self)

    def __add__(self, other):
        return _Add(self, other)

    def __mul__(self, other):
        return _Mul(self, other)

    def __pow__(self, other):
        return _Pow(self, other)

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return _Mul(self, other)

    def __truediv__(self, other):  # self / other
        return _Div(self, other)

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        grad_fn = f', grad_fn={self.grad_fn}' if self.grad_fn else ''
        return f"Value(data={round(self.data, 4)}, " +\
            f"grad={round(self.grad, 4)}{grad_fn})"
