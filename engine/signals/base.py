"""
engine/signals/base.py — Interface commune pour tous les agents de signal.

Chaque agent hérite de BaseSignal et implémente generate().
Le contrat est identique pour tous les agents : même entrées, même sorties.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseSignal(ABC):
    """Interface commune pour les générateurs de signaux."""

    def __init__(self, params: dict = None):
        self.params = {**self.default_params, **(params or {})}

    @property
    def default_params(self) -> dict:
        """Paramètres par défaut spécifiques au signal."""
        return {}

    @abstractmethod
    def generate(self, df: pd.DataFrame, deriv_data: dict = None) -> pd.DataFrame:
        """
        Génère les signaux sur un DataFrame OHLCV enrichi d'indicateurs.

        Entrée :
            df          — DataFrame avec colonnes OHLCV + indicateurs calculés
                          (vol_realized, vol_annualized, zscore, atr, regime, ...)
            deriv_data  — dict optionnel retourné par fetch_all_derivatives()

        Sortie :
            DataFrame identique + colonnes ajoutées :
                signal        int    +1 (long) / -1 (short) / 0 (neutre)
                signal_raw    float  valeur brute ayant déclenché le signal
                signal_label  str    description lisible ("LONG z<-2.0", ...)
        """

    def param_summary(self) -> str:
        """Représentation lisible des paramètres actifs."""
        return ", ".join(f"{k}={v}" for k, v in self.params.items())

    @property
    def name(self) -> str:
        """Nom du signal."""
        return self.__class__.__name__.lower()

    @property
    def description(self) -> str:
        """Description brève du signal."""
        return ""
