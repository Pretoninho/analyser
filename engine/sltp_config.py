"""
engine/sltp_config.py — Gestion adaptative SL/TP par volatilité et R:R.

Permet de paramétrer le Risk:Reward ratio et d'ajuster automatiquement
les niveaux de SL/TP en fonction du régime de volatilité.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SLTPConfig:
    """Configuration de Stop Loss / Take Profit."""
    
    # Risk:Reward ratio (ex: 2.0 → TP = SL × 2)
    risk_reward_ratio: float = 2.0
    
    # Base SL en % par régime de volatilité
    sl_low: float = 0.003      # 0.3%
    sl_rising: float = 0.003   # 0.3%
    sl_high: float = 0.004     # 0.4%
    sl_extreme: float = 0.006  # 0.6%
    sl_falling: float = 0.003  # 0.3%
    
    def get_sl_tp(self, volatility_level: int = 0) -> Tuple[float, float]:
        """
        Retourne (SL%, TP%) pour un niveau de volatilité donné.
        
        volatility_level:
            0 = LOW
            1 = RISING
            2 = HIGH
            3 = EXTREME
            4 = FALLING
        """
        sl_map = {
            0: self.sl_low,
            1: self.sl_rising,
            2: self.sl_high,
            3: self.sl_extreme,
            4: self.sl_falling,
        }
        sl = sl_map.get(volatility_level, self.sl_low)
        tp = sl * self.risk_reward_ratio
        return (sl, tp)
    
    def get_sl_tp_from_regime(self, regime: str) -> Tuple[float, float]:
        """
        Retourne (SL%, TP%) pour un régime nommé.
        regime: "LOW", "RISING", "HIGH", "EXTREME", "FALLING"
        """
        regime_map = {
            "LOW": 0,
            "RISING": 1,
            "HIGH": 2,
            "EXTREME": 3,
            "FALLING": 4,
        }
        level = regime_map.get(regime, 0)
        return self.get_sl_tp(level)
    
    def __str__(self) -> str:
        return (
            f"SLTPConfig(R:R={self.risk_reward_ratio}, "
            f"SL_low={self.sl_low*100:.1f}%, SL_high={self.sl_high*100:.1f}%)"
        )


# Config par défaut (recommandée pour production)
DEFAULT_CONFIG = SLTPConfig(
    risk_reward_ratio=2.0,
    sl_low=0.003,
    sl_rising=0.003,
    sl_high=0.004,
    sl_extreme=0.006,
    sl_falling=0.003,
)
