use nalgebra::DMatrix;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct Dataset {
    pub num_clusters: usize,
    pub num_samples: usize,
    pub num_features: usize,
    pub true_labels: Vec<usize>,
    pub data: DMatrix<f64>,
}

#[derive(PartialEq, Eq, Hash)]
pub enum DatasetType {
    Iris,
    BreastCancer,
    Wine,
    Glass,
    Seeds,
}



impl DatasetType {
    pub fn to_string(&self) -> String {
        match self {
            DatasetType::Iris => "iris".to_string(),
            DatasetType::BreastCancer => "breast_cancer".to_string(),
            DatasetType::Wine => "wine".to_string(),
            DatasetType::Glass => "glass".to_string(),
            DatasetType::Seeds => "seeds".to_string(),
        }
    }

    pub fn from_int(i: usize) -> Self {
        match i {
            0 => DatasetType::Iris,
            1 => DatasetType::BreastCancer,
            2 => DatasetType::Wine,
            3 => DatasetType::Glass,
            4 => DatasetType::Seeds,
            _ => panic!("Invalid dataset type"),
        }
    }
}

impl Dataset {
    fn split(line: &str, delimiter: char) -> Vec<String> {
        line.split(delimiter).map(|s| s.to_string()).collect()
    }

    pub fn load_from_csv(filename: &str, normalize: bool) -> Result<Self, Box<dyn Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        let mut lines = reader.lines();
        let num_clusters = lines.next().ok_or("Missing number of clusters")??.parse()?;

        let mut data = Vec::new();
        let mut true_labels = Vec::new();

        for line in lines {
            let line = line?;
            let tokens = Self::split(&line, ',');
            if tokens.is_empty() {
                continue;
            }

            let features: Vec<f64> = tokens[..tokens.len() - 1]
                .iter()
                .map(|s| s.parse().unwrap())
                .collect();
            let label = tokens.last().unwrap().parse().unwrap();

            true_labels.push(label);
            data.push(features);
        }

        let num_samples = data.len();
        let num_features = data[0].len();

        if normalize {
            for j in 0..num_features {
                let mut min_val = data[0][j];
                let mut max_val = data[0][j];

                for i in 1..num_samples {
                    min_val = min_val.min(data[i][j]);
                    max_val = max_val.max(data[i][j]);
                }

                let range = max_val - min_val;
                if range > 0.0 {
                    for i in 0..num_samples {
                        data[i][j] = (data[i][j] - min_val) / range;
                    }
                }
            }
        }

        let mut matrix_data = Vec::new();
        for row in data {
            matrix_data.extend(row);
        }

        let data_matrix = DMatrix::from_vec(num_samples, num_features, matrix_data);

        Ok(Dataset {
            num_clusters,
            num_samples,
            num_features,
            true_labels,
            data: data_matrix,
        })
    }
}
