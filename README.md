# algoFuzz

[![Documentation](https://img.shields.io/readthedocs/algofuzz)](https://algofuzz.naghi.me)

[![PyPI version](https://img.shields.io/pypi/v/algofuzz.svg)](https://pypi.python.org/pypi/algofuzz/)

algoFuzz is a unique framework dedicated to the implementation and comparison of various fuzzy c-means clustering algorithms found in academic literature. It provides a comprehensive environment for researchers and practitioners to explore and compare different clustering methodologies.

# Documentation

The full documentation for algoFuzz is available at [Read the Docs](https://algofuzz.naghi.me). You should read the documentation before using algoFuzz.

# Getting Started

Before you start using the algoFuzz library, ensure that you have the following prerequisites:

- Python 3.8+ installed on your system.
- Familiarity with clustering concepts.
- A basic understanding of Python programming.

To install the library, simply use the following pip command:

```bash
pip install algofuzz
```

Alternatively, you can also install the library directly from the source if it’s hosted on GitHub or a similar platform:

```bash
git clone https://github.com/naghim/algofuzz
cd algofuzz
pip install .
```

# Quick example

Here’s a basic example of how to use the Fuzzy C-Means (FCM) algorithm to cluster data.

Refer to the [documentation](https://algofuzz.naghi.me/en/latest/getting_started.html) for a thorough explanation.

```python
from algofuzz import FCM
from algofuzz import CentroidStrategy, DatasetType, load_dataset, generate_colors
import numpy as np
import matplotlib.pyplot as plt
import random

# Choose a random seed for reproducibility
np.random.seed(0)

# Load the Bubbles toy dataset
# You can replace this with your own dataset as well
data, num_clusters, true_labels = load_dataset(DatasetType.Bubbles)

# There are 3 clusters
print(f'Number of clusters: {num_clusters}')

# Create an FCM model with 3 clusters, choosing random initial centroids
fcm = FCM(
   num_clusters=num_clusters,
   max_iter=100,
   centroid_strategy=CentroidStrategy.Random
)

# Fit the model to the data
fcm.fit(data)

# These are the centroids of the clusters
centers = fcm.centroids

# These are the labels assigned to each data point (there are 3 clusters)
labels = fcm.labels
```

# Algorithms

You may choose from the following algorithms currently available in the library:

1. Fuzzy C-Means (FCM) algorithm, proposed by Dunn in 1973 and improved by Bezdek in 1981.

- `from algofuzz import FCM`

2. Possibilistic Fuzzy C-Means Clustering algorithm proposed by Pal et al. in 2005.

- `from algofuzz import PFCM`

3. An extension of the FCM model that includes a penalty term (eta) for each cluster depending on the distance between the data points and the centroids of the clusters.

- `from algofuzz import EtaFCM`

4. Fuzzy C-Means algorithm with an extra noise cluster proposed by R. Dave in 1993.

- `from algofuzz import FCPlus1M`

5. Fuzzy Possibilistic Product Partition C-Means Clustering algorithm proposed by L. Szilágyi & S. Szilágyi in 2014.

- `from algofuzz import NonOptimizedFP3CM`

6. Fuzzy-Possibilistic C-Means Clustering algorithm proposed by Pal, Pal and Bezdek in 1997.

- `from algofuzz import NonOptimizedFPCM`

7. Self-tuning version of the Possibilistic Fuzzy C-Means Clustering algorithm proposed by MB. Naghi in 2023.

- `from algofuzz import STPFCM`
