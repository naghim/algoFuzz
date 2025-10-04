#ifndef FPCM_H
#define FPCM_H

#include "GFPCM.h"
#include <unordered_map>

/**
 * @brief Partitions a numeric dataset using the Fuzzy-Possibilistic C-Means Clustering (FPCM) algorithm.
 *        This is a derivative of GFPCM with w_prob always set to 1.0.
 */
class FPCM : public GFPCM
{
public:
    FPCM(int num_clusters, int max_iter = 150, float m = 2.0f, float p = 2.0f, float noise = 0.0f);
};

#endif // FPCM_H