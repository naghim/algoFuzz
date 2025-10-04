from enum import Enum

__all__ = ['DatasetType', 'FCMType', 'CentroidStrategy']

DatasetType = Enum('DatasetType', [
    'Iris', 'Glass', 'Seeds', 'NormalizedIris', 'NormalizedGlass', 'NormalizedSeeds',
    'NoisyNormalizedIris', 'Bubbles', 'PrevBubbles', 'Wine', 'NormalizedWine', 'BreastCancer',
    'NormalizedBreastCancer', 'Bubbles1', 'Bubbles2', 'Bubbles3', 'Bubbles4', 'Spellman',
    'NormalizedSpellman',
    'NormalizedNoisyNIris1', 'NormalizedNoisyNIris5', 'NormalizedNoisyNIris10',
    'NormalizedNoisyNIris20', 'NormalizedNoisyNIris50', 'NormalizedNoisyNIris100'
])
FCMType = Enum('FCMType', ['FCM', 'FCPlus1M', 'STPFCM', 'PFCM', 'FP3CM', 'FPCM', 'GFPCM'])
CentroidStrategy = Enum('CentroidStrategy', ['Random', 'FixedRangeOutliers', 'Outliers', 'Sample', 'Diagonal', 'NormalizedIrisDiagonal', 'NormalizedBreastDiagonal', 'Mirtill' ,'Custom'])
