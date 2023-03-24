from ._classification import Hinge
from ._classification import Softmax
from ._classification import Squared_hinge

from ._regression import MAE
from ._regression import MSE
from ._regression import MAPE
from ._regression import Huber

__all__ = [
    'Hinge',
    'Softmax',
    'Squared_hinge',

    'MAE',
    'MSE',
    'MAPE',
    'Huber'
]
