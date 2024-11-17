from enum import Enum

__all__ = ['DatasetType', 'FCMType', 'CentroidStrategy']

DatasetType = Enum('DatasetType', ['Iris', 'Glass', 'Seeds', 'NormalizedIris', 'NormalizedGlass', 'NormalizedSeeds', 'NoisyNormalizedIris',
                   'Bubbles', 'PrevBubbles', 'Wine', 'NormalizedWine', 'BreastCancer', 'NormalizedBreastCancer', 'Bubbles1', 'Bubbles2', 'Bubbles3', 'Bubbles4'])
FCMType = Enum('FCMType', ['FCM', 'FCPlus1M', 'NonoptimizedSTPFCM', 'STPFCM',
               'PFCM', 'NonoptimizedFP3CM', 'NonoptimizedFPCM', 'NonoptimizedGFPCM'])
CentroidStrategy = Enum('CentroidStrategy', [
                        'Random', 'Outliers', 'Sample', 'Diagonal', 'NormalizedIrisDiagonal', 'NormalizedBreastDiagonal', 'Custom'])
