__all__ = ['EtaFCM', 'FCM', 'FCPlus1M', 'NonoptimizedFPCM', 'NonoptimizedFP3CM', 'PFCM', 'STPFCM']

from .base_fcm import BaseFCM
from .eta_fcm import EtaFCM
from .fcm import FCM
from .fcplus1m import FCPlus1M
from .nonoptimized_fp3cm import NonoptimizedFP3CM
from .nonoptimized_fpcm import NonoptimizedFPCM
from .possibilistic_fcm import PFCM
from .stpfcm import STPFCM
from algofuzz.enums import FCMType

def get_fcm_by_type(fcm_type: FCMType | str) -> BaseFCM:
    if isinstance(fcm_type, str):
        fcm_type = FCMType[fcm_type]

    if fcm_type == FCMType.FCM:
        return FCM
    elif fcm_type == FCMType.FCPlus1M:
        return FCPlus1M
    elif fcm_type == FCMType.STPFCM:
        return STPFCM
    elif fcm_type == FCMType.PFCM:
        return PFCM
    elif fcm_type == FCMType.NonoptimizedFP3CM:
        return NonoptimizedFP3CM
    elif fcm_type == FCMType.NonoptimizedFPCM:
        return NonoptimizedFPCM
