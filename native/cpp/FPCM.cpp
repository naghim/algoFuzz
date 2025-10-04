#include "FPCM.h"

FPCM::FPCM(int num_clusters, int max_iter, float m, float p, float noise)
    : GFPCM(num_clusters, max_iter, m, p, 1.0f, noise) { // w_prob is always 1.0 for FPCM
    // No additional initialization needed as GFPCM constructor handles it
}