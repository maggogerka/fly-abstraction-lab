"""Model families with a shared multitask interface."""

from fly_abstraction.models.baselines import GRUBaseline, MLPBaseline
from fly_abstraction.models.connectome import (
    FixedConnectomeReservoir,
    RandomGraphReservoir,
    TrainableConnectomeRNN,
)

__all__ = [
    "FixedConnectomeReservoir",
    "GRUBaseline",
    "MLPBaseline",
    "RandomGraphReservoir",
    "TrainableConnectomeRNN",
]
