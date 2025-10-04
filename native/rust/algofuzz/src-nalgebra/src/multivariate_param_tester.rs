use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use crate::centroid_strategy::CentroidStrategy;
use crate::dataset_loader::{Dataset, DatasetType};
use crate::gfpcm::GFPCM;
use crate::metrics::{calculate_ari, calculate_nmi, calculate_purity};

pub struct MultivariateParamTester {
    param_grid: HashMap<String, Vec<f64>>,
    datasets: HashMap<DatasetType, Arc<Dataset>>,
}

impl MultivariateParamTester {
    pub fn new(param_grid: HashMap<String, Vec<f64>>) -> Self {
        Self {
            param_grid,
            datasets: HashMap::new(),
        }
    }

    pub fn queue_options(&self) -> Vec<Vec<f64>> {
        let mut process_queue = Vec::new();

        for &dataset in &self.param_grid["dataset"] {
            for &m in &self.param_grid["m"] {
                for &p in &self.param_grid["p"] {
                    for &seed in &self.param_grid["seed"] {
                        for &w_prob in &self.param_grid["w_prob"] {
                            for &noise in &self.param_grid["noise"] {
                                for &centroid_strategy in &self.param_grid["centroid_strategy"] {
                                    let actual_w_prob = 1000.0;
                                    println!(
                                        "Dataset: {}, m: {}, p: {}, seed: {}, w_prob: {}, noise: {}, centroid_strategy: {}",
                                        dataset, m, p, seed, actual_w_prob, noise, centroid_strategy
                                    );
                                    process_queue.push(vec![
                                        dataset,
                                        m,
                                        p,
                                        seed,
                                        actual_w_prob,
                                        noise,
                                        centroid_strategy,
                                    ]);
                                }
                            }
                        }
                    }
                }
            }
        }

        println!(
            "Total number of parameter combinations: {}",
            process_queue.len()
        );
        process_queue
    }

    pub fn evaluate(dataset: &Arc<Dataset>, item: &Vec<f64>) -> Vec<f64> {
        //println!("Evaluating: {:?}", item);
        let max_iter = 150;
        let m = item[1];
        let p = item[2];
        let w_prob = item[4];
        // TODO
        //let noise = item[5];
        let centroid_strategy = CentroidStrategy::to_centroid_strategy(item[6]);

        let mut fcm = GFPCM::new(
            dataset.num_clusters,
            max_iter,
            m,
            p,
            w_prob,
            centroid_strategy,
        );

        let start = Instant::now();
        fcm.fit(&dataset.data, &dataset.true_labels);
        let duration = start.elapsed().as_secs_f64();

        let labels = fcm.predicted_labels;
        let purity = calculate_purity(&dataset.true_labels, &labels);
        //let nmi = calculate_nmi(&dataset.true_labels, &labels);
        //let ari = calculate_ari(&dataset.true_labels, &labels);

        let mut result = item.clone();
        result.push(duration);
        result.push(purity);
        //result.push(nmi);
        //result.push(ari);

        result
    }

    pub fn fit_to_csv(&self, filename: &str) {
        let process_queue = self.queue_options();

        println!("filename now: {}", filename);
        let file = File::create(filename).unwrap();
        let mut writer = std::io::BufWriter::new(file);

        writeln!(
            writer,
            "dataset,m,p,seed,w_prob,noise,centroid_strategy,time,purity" //,nmi,ari"
        )
        .unwrap();

        //let (tx, rx) = mpsc::channel();
        let chunk_size = 1000;
        let num_chunks = (process_queue.len() + chunk_size - 1) / chunk_size;

        println!("Number of chunks: {}", num_chunks);

        let n_workers = num_cpus::get();
        println!("Made this many workers: {}", n_workers);
        let pool = threadpool::ThreadPool::new(n_workers);
        let (tx, rx) = mpsc::channel();

        let start_time = Instant::now();

        for chunk in 0..num_chunks {
            let start_idx = chunk * chunk_size;
            let end_idx = std::cmp::min(start_idx + chunk_size, process_queue.len());
            let chunk_queue = process_queue[start_idx..end_idx].to_vec();
            let count = chunk_queue.len();

            for item in chunk_queue {
                let tx = tx.clone();
                let dataset = self.datasets[&DatasetType::from_int(item[0] as usize)].clone();

                pool.execute(move || {
                    let results = MultivariateParamTester::evaluate(&dataset, &item);
                    tx.send(results).unwrap();
                });
            }

            for i in 0..count {
                let result = rx.recv().unwrap();
                if !result.is_empty() {
                    for i in 0..result.len() - 1 {
                        write!(writer, "{},", result[i]).unwrap();
                    }
                    writeln!(writer, "{}", result.last().unwrap()).unwrap();
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        println!("Total time required: {} seconds", total_time);
    }

    pub fn load_datasets(&mut self) {
        self.datasets.insert(
            DatasetType::Iris,
            Arc::new(Dataset::load_from_csv("iris.csv", true).ok().unwrap()),
        );
        self.datasets.insert(
            DatasetType::BreastCancer,
            Arc::new(
                Dataset::load_from_csv("breast_cancer.csv", true)
                    .ok()
                    .unwrap(),
            ),
        );
        self.datasets.insert(
            DatasetType::Wine,
            Arc::new(Dataset::load_from_csv("wine.csv", true).ok().unwrap()),
        );
        self.datasets.insert(
            DatasetType::Glass,
            Arc::new(Dataset::load_from_csv("glass.csv", true).ok().unwrap()),
        );
        self.datasets.insert(
            DatasetType::Seeds,
            Arc::new(Dataset::load_from_csv("seeds.csv", true).ok().unwrap()),
        );
    }
}
