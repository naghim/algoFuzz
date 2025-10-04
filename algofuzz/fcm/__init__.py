__all__ = ['EtaFCM', 'FCM', 'FCPlus1M', 'FPCM', 'GFPCM', 'FP3CM', 'PFCM', 'STPFCM']

from algofuzz._algofuzz import BaseFCM, EtaFCM, FCM, FCPlus1M, FPCM, GFPCM, FP3CM, PFCM, STPFCM
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
    elif fcm_type == FCMType.FP3CM:
        return FP3CM
    elif fcm_type == FCMType.FPCM:
        return FPCM
    elif fcm_type == FCMType.GFPCM:
        return GFPCM