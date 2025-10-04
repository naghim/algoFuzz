from enum import Enum

__all__ = ['DatasetType', 'FCMType', 'CentroidStrategy']

DatasetType = Enum('DatasetType', ['Iris', 'Glass', 'Seeds', 'NormalizedIris', 'NormalizedGlass', 'NormalizedSeeds', 'NoisyNormalizedIris', 'Bubbles', 'PrevBubbles', 'Wine', 'NormalizedWine', 'BreastCancer', 'NormalizedBreastCancer', 'Bubbles1', 'Bubbles2', 'Bubbles3', 'Bubbles4', 'Spellman', 'NormalizedSpellman'])
FCMType = Enum('FCMType', ['FCM', 'FCPlus1M', 'STPFCM', 'PFCM', 'FP3CM', 'FPCM', 'GFPCM'])
CentroidStrategy = Enum('CentroidStrategy', ['Random', 'Outliers', 'Sample', 'Diagonal', 'NormalizedIrisDiagonal', 'NormalizedBreastDiagonal', 'Mirtill' ,'Custom'])
