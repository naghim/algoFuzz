===============
Getting Started
===============

Welcome to algoFuzz, a framework for popular fuzzy c-means clustering algorithms from literature. This guide will walk you through the basics of using algoFuzz, from installation to running your first clustering algorithm.

Prerequisites
---------------------------------
Before you start using the algoFuzz library, ensure that you have the following prerequisites:

- **Python 3.7+** installed on your system.
- Familiarity with clustering concepts.
- A basic understanding of Python programming.

Installation
---------------------------------
To install the library, simply use the following ``pip`` command:

.. code-block:: bash

   pip install algofuzz

Alternatively, you can also install the library directly from the source if it's hosted on GitHub or a similar platform:

.. code-block:: bash

   git clone https://github.com/naghim/algofuzz
   cd algofuzz
   pip install .

Make sure to check the `official repository <https://github.com/naghim/algofuzz>`_ for the latest updates and releases.

Quick Example
---------------------------------

Here's a basic example of how to use the Fuzzy C-Means (FCM) algorithm to cluster data.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from algofuzz import FCM
    
    # Generate random data points
    np.random.seed(0)
    data = np.random.rand(100, 2)
    
    # Create an FCM model with 3 clusters
    fcm = FCM(n_clusters=3, max_iter=100)
    fcm.fit(data)
    
    # Get cluster centers
    centers = fcm.get_centers()
    
    # Plot the clusters
    plt.scatter(data[:, 0], data[:, 1], c=fcm.predict(data))
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
    plt.show()

Explanation
-----------

- **Step 1:** We import the necessary modules, including FCM from the algoFuzz library.
- **Step 2:** We generate some random data points for clustering.
- **Step 3:** We create an FCM object, specifying the number of clusters (in this case, 3).
- **Step 4:** The `fit` method is used to compute clusters and membership values based on the input data.
- **Step 5:** Finally, we visualize the clusters using a scatter plot.

Parameters
==========

- **n_clusters**: (int) The number of clusters you want to segment the data into.
- **max_iter**: (int) Maximum number of iterations to run the FCM algorithm. Default is 150.
- **m**: (float) The fuzziness coefficient. Must be greater than 1. Default is 2.0.
- **error**: (float) The stopping criterion for the algorithm. Default is 1e-5.

Example:

.. code-block:: python

   fcm = FCM(n_clusters=4, m=1.5, max_iter=200, error=1e-4)

Advanced Usage
==============

You can customize the behavior of the algorithms by tuning additional parameters, such as the fuzziness coefficient ``m`` or the maximum number of iterations. You can also set your own initial cluster centers using the ``centroids`` parameter or set the stra  initializing the cluster centers using the ``centroid_strategy`` parameter. access the cluster centers and membership values after fitting the model. 

Here is how to do it in case of FCM:

Example:

.. code-block:: python

   # Custom FCM model
   fcm = FCM(n_clusters=5, m=2.5, max_iter=300)
   fcm.fit(data)

   # Get cluster centers
   centers = fcm.get_centers()
   print("Cluster Centers:", centers)

Further Reading
===============

Visit the `Theoretical background <https://en.wikipedia.org/wiki/Fuzzy_clustering>`_ page for more information about this topic.

Need Help?
==========

If you encounter any issues or need further clarification, feel free to reach out via:

- `GitHub Issues <https://github.com/naghim/algofuzz/issues>`_ for bug reports and feature requests.
- `Discord Community Server <https://discord.gg/7rDajmdEPV>`_ for discussions and troubleshooting.
