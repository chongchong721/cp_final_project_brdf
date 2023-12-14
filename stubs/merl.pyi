from __future__ import annotations
__all__ = ['MERL_BRDF']
class MERL_BRDF:
    b_channel_unscaled: list[float]
    g_channel_unscaled: list[float]
    m_data: float
    r_channel_unscaled: list[float]
    def __init__(self, arg0: str) -> None:
        ...
    def convert_to_hd(self, arg0: float, arg1: float, arg2: float, arg3: float) -> list[float]:
        ...
    def look_up(self, arg0: float, arg1: float, arg2: float, arg3: float) -> list[float]:
        ...
    def reduce_phi_d(self, arg0: float) -> float:
        ...
    @property
    def b_scale(self) -> float:
        ...
    @property
    def g_scale(self) -> float:
        ...
    @property
    def m_size(self) -> int:
        ...
    @property
    def r_scale(self) -> float:
        ...
