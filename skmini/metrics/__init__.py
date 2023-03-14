from ._classification import BCE
from ._classification import Hinge
from ._classification import Squared_hinge

from ._regression import MAE
from ._regression import MSE
from ._regression import MAPE
from ._regression import Huber

__all__ = [
    'BCE',
    'Hinge',
    'Squared_hinge',

    'MAE',
    'MSE',
    'MAPE',
    'Huber'
]
