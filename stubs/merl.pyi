from __future__ import annotations
__all__ = ['MERL_BRDF', 'convert_to_hd', 'get_half_diff_coord_from_index', 'get_half_diff_idxes_from_index', 'get_index_from_half_diff_idxes', 'get_index_from_hall_diff_coords', 'phi_diff_rad', 'reduce_phi_d', 'theta_diff_rad', 'theta_half_rad']
class MERL_BRDF:
    b_channel_unscaled: list[float]
    g_channel_unscaled: list[float]
    r_channel_unscaled: list[float]
    def __init__(self, arg0: str) -> None:
        ...
    def look_up(self, arg0: float, arg1: float, arg2: float, arg3: float) -> list[float]:
        ...
    def look_up_channel(self, arg0: float, arg1: float, arg2: float, arg3: float, arg4: int) -> float:
        ...
    def look_up_hdidx(self, arg0: float, arg1: float, arg2: float) -> list[float]:
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
def convert_to_hd(arg0: float, arg1: float, arg2: float, arg3: float) -> list[float]:
    """
    params -> [0]:theta_in [1]:theta_in [2]theta_out [3]phi_out
    return list[float] -> [0]:theta_half [1]:phi_half [2]:theta_diff [3]phi_diff
    """
def get_half_diff_coord_from_index(arg0: int) -> list[float]:
    """
    return list[float] in rad -> [0]:theta_half [1]:theta_diff [2]phi_diff
    """
def get_half_diff_idxes_from_index(arg0: int) -> list[int]:
    """
    return list[int] in degree/index -> [0]:theta_half [1]:theta_diff [2]phi_diff
    """
def get_index_from_half_diff_idxes(arg0: int, arg1: int, arg2: int) -> int:
    """
    param -> [0]:theta_half [1]:theta_diff [2]phi_diff
    """
def get_index_from_hall_diff_coords(arg0: float, arg1: float, arg2: float) -> int:
    """
    params -> [0]:theta_half [1]:theta_diff [2]phi_diff
    """
def phi_diff_rad(arg0: int) -> float:
    ...
def reduce_phi_d(arg0: float) -> float:
    ...
def theta_diff_rad(arg0: int) -> float:
    ...
def theta_half_rad(arg0: int) -> float:
    ...
