use std::collections::HashMap;
use multivariate_param_tester::MultivariateParamTester;

mod centroid_strategy;
mod dataset_loader;
mod gfpcm;
mod metrics;
mod multivariate_param_tester;

fn main() {
    let only_one = false;

    let mut param_grid: HashMap<String, Vec<f64>> = HashMap::new();
    if !only_one {
        param_grid.insert("dataset".to_string(), vec![1.0, 2.0, 3.0, 4.0]); // Example dataset types
        param_grid.insert("m".to_string(), vec![1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]);
        param_grid.insert("p".to_string(), vec![2.0]);
        param_grid.insert("seed".to_string(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        param_grid.insert("w_prob".to_string(), vec![-1.0]);
        param_grid.insert("noise".to_string(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]);
        param_grid.insert("centroid_strategy".to_string(), vec![0.0]); // Example centroid strategy
    } else {
        param_grid.insert("dataset".to_string(), vec![0.0]); // Example dataset type
        param_grid.insert("m".to_string(), vec![2.0]);
        param_grid.insert("p".to_string(), vec![2.0]);
        param_grid.insert("seed".to_string(), vec![0.0]);
        param_grid.insert("w_prob".to_string(), vec![1.0]);
        param_grid.insert("noise".to_string(), vec![0.0]);
        param_grid.insert("centroid_strategy".to_string(), vec![0.0]); // Example centroid strategy
    }

    let mut tester = MultivariateParamTester::new(param_grid);
    tester.load_datasets();
    tester.fit_to_csv("GFPCM_res_cpp.csv");
}