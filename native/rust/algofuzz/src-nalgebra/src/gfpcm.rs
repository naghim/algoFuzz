use nalgebra::{DMatrix, DVector};
use crate::centroid_strategy::{CentroidStrategy, CentroidUtility};

pub struct GFPCM {
    num_clusters: usize,
    max_iter: usize,
    m: f64,
    p: f64,
    w_prob: f64,
    centroids: DMatrix<f64>,
    centroid_strategy: CentroidStrategy,
    pub trained: bool,
    eta: DMatrix<f64>,
    pub member: DMatrix<f64>,
    pub predicted_labels: Vec<usize>,
}

impl GFPCM {
    pub fn new(num_clusters: usize, max_iter: usize, m: f64, p: f64, w_prob: f64, centroid_strategy: CentroidStrategy) -> GFPCM {
        GFPCM {
            num_clusters,
            max_iter,
            m,
            p,
            w_prob,
            centroids: DMatrix::zeros(1, 1),
            trained: false,
            eta: DMatrix::zeros(1, 1),
            member: DMatrix::zeros(1, 1),
            predicted_labels: Vec::new(),
            centroid_strategy,
        }
    }

    pub fn fit(&mut self, X: &DMatrix<f64>, true_labels: &Vec<usize>) {
        let actual_n = X.ncols();

        let z = X.nrows();
        let n = X.ncols();

        let mut u = DMatrix::zeros(self.num_clusters, n);
        let mut t = DMatrix::zeros(self.num_clusters, n);

        let deriv_m = -2.0 / (self.m - 1.0);
        let deriv_p = -2.0 / (self.p - 1.0);

        // Initialize centroids (assuming a function initialize_clusters exists)
        self.centroids = DMatrix::zeros(z, self.num_clusters);
        CentroidUtility::initialize_clusters(&X, &mut self.centroids, self.num_clusters, &self.centroid_strategy);

        for _ in 0..self.max_iter {
            // Update u
            for k in 0..n {
                let mut szum = 0.0;
                for i in 0..self.num_clusters {
                    u[(i, k)] = (X.column(k) - self.centroids.column(i))
                        .norm()
                        .powf(deriv_m);
                    szum += u[(i, k)];
                }
                for i in 0..self.num_clusters {
                    u[(i, k)] /= szum;
                }
            }

            // Update t
            for i in 0..self.num_clusters {
                let mut szum = 0.0;
                for k in 0..n {
                    t[(i, k)] = (X.column(k) - self.centroids.column(i))
                        .norm()
                        .powf(deriv_p);
                    szum += t[(i, k)];
                }
                for k in 0..n {
                    t[(i, k)] /= szum;
                }
            }

            // Update centroids
            for i in 0..self.num_clusters {
                let mut sumup = DVector::zeros(z);
                let mut sumdn = 0.0;
                for k in 0..n {
                    let weight = u[(i, k)].powf(self.m) + self.w_prob * t[(i, k)].powf(self.p);
                    sumup += weight * X.column(k);
                    sumdn += weight;
                }
                self.centroids.set_column(i, &(sumup / sumdn));
            }
        }

        self.eta = DMatrix::zeros(self.num_clusters, n);
        self.member = (u.map(|v| v.powf(self.m)) + t.map(|v| v.powf(self.p)) * self.w_prob).clone();
        self.trained = true;

        //println!("{:?}", self.member);

        self.predicted_labels = Vec::new();

        for k in 0..actual_n {
            let mut best_cluster = 0;
            let mut max_membership = 0.0;

            // Find the cluster with the highest membership value
            for i in 0..self.num_clusters {
                if self.member[(i, k)] > max_membership {
                    max_membership = self.member[(i, k)];
                    best_cluster = i;
                }
            }

            self.predicted_labels.push(best_cluster);
        }

        // Compute confusion matrix (assuming a function compute_confusion_matrix exists)
        let confusion_matrix = self.compute_confusion_matrix(true_labels, &self.predicted_labels);
        //println!("Confusion matrix: {:?}", confusion_matrix);
    }

    fn compute_confusion_matrix(
        &self,
        _true_labels: &Vec<usize>,
        _predicted_labels: &Vec<usize>,
    ) -> DMatrix<f64> {
        // Implementation for computing confusion matrix
        DMatrix::zeros(1, 1)
    }
}

/*
fn main() {
    let num_clusters = 3;
    let max_iter = 100;
    let m = 2.0;
    let p = 2.0;
    let w_prob = 0.5;
    let noise = 0.0;

    let X = DMatrix::from_row_slice(4, 4, &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]);

    let true_labels = vec![0, 1, 2, 0];

    let mut gfpcm = GFPCM {
        num_clusters,
        max_iter,
        m,
        p,
        w_prob,
        noise,
        centroids: DMatrix::zeros(4, num_clusters),
        trained: false,
        eta: DMatrix::zeros(num_clusters, X.ncols()),
        member: DMatrix::zeros(num_clusters, X.ncols()),
    };

    gfpcm.fit(&X, &true_labels);
}*/