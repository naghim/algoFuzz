use nalgebra::{DMatrix, DVector};

pub enum CentroidStrategy {
    Mirtill,
    IrisDiagonal,
    KMeans,
}

impl CentroidStrategy {
    pub fn to_centroid_strategy(strategy: f64) -> CentroidStrategy {
        match strategy as i32 {
            0 => CentroidStrategy::Mirtill,
            1 => CentroidStrategy::IrisDiagonal,
            2 => CentroidStrategy::KMeans,
            _ => panic!("Invalid centroid strategy"),
        }
    }
}

pub struct CentroidUtility;

impl CentroidUtility {
    pub fn initialize_clusters(
        X: &DMatrix<f64>,
        centroids: &mut DMatrix<f64>,
        num_clusters: usize,
        cluster_strategy: &CentroidStrategy,
    ) {
        *centroids = DMatrix::zeros(X.nrows(), num_clusters);

        match cluster_strategy {
            CentroidStrategy::Mirtill => {
                for d in 0..num_clusters {
                    let val = d as f64 / (num_clusters - 1) as f64;
                    centroids.set_column(d, &DVector::from_element(X.nrows(), val));
                }
            }
            CentroidStrategy::IrisDiagonal => {
                for d in 0..X.nrows() {
                    centroids[(d, 0)] = (1.0 - (-1.0f64).powi(d as i32)) / 2.0;
                    centroids[(d, 1)] = 0.5;
                    centroids[(d, 2)] = (1.0 + (-1.0f64).powi(d as i32)) / 2.0;
                }
            }
            CentroidStrategy::KMeans => {
                // Not implemented
            }
        }

        let debug = false;

        if debug {
            println!("Centroids matrix:\n{}", centroids);
        }
    }
}
