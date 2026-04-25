"""engine/signals — Signal generation strategies."""

from .base import BaseSignal
from .zscore_signal import ZScoreSignal
from .liquidation_signal import LiquidationSignal
from .combined_signal import CombinedSignal

SIGNAL_REGISTRY = {
    "zscore": ZScoreSignal,
    "liquidation": LiquidationSignal,
    "combined": CombinedSignal,
}

__all__ = [
    "BaseSignal",
    "ZScoreSignal",
    "LiquidationSignal",
    "CombinedSignal",
    "SIGNAL_REGISTRY",
]
