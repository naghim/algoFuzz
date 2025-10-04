use nalgebra::DMatrix;
use std::collections::HashMap;
use std::f64::consts::LOG2_E;

pub fn calculate_purity(true_labels: &[usize], predicted_labels: &[usize]) -> f64 {
    let mut contingency_table: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (&true_label, &predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        *contingency_table.entry(predicted_label).or_default().entry(true_label).or_default() += 1;
    }

    let total_count = true_labels.len();
    let mut correct_count = 0;

    for cluster in contingency_table.values() {
        let max_count = cluster.values().max().unwrap_or(&0);
        correct_count += max_count;
    }

    correct_count as f64 / total_count as f64
}

fn entropy(labels: &[usize]) -> f64 {
    let mut label_counts: HashMap<usize, usize> = HashMap::new();
    for &label in labels {
        *label_counts.entry(label).or_default() += 1;
    }

    let total_count = labels.len() as f64;
    label_counts.values().fold(0.0, |ent, &count| {
        let p = count as f64 / total_count;
        ent - p * (p * LOG2_E).ln()
    })
}

fn calculate_mutual_information(true_labels: &[usize], predicted_labels: &[usize]) -> f64 {
    let mut contingency_table: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (&true_label, &predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        *contingency_table.entry(predicted_label).or_default().entry(true_label).or_default() += 1;
    }

    let total_count = true_labels.len() as f64;
    let mut mutual_information = 0.0;

    for cluster in contingency_table.values() {
        let cluster_size: usize = cluster.values().sum();
        for (&label, &nij) in cluster {
            if nij > 0 {
                let pij = nij as f64 / total_count;
                let pi = cluster_size as f64 / total_count;
                let pj = true_labels.iter().filter(|&&l| l == label).count() as f64 / total_count;
                mutual_information += pij * (pij / (pi * pj)).ln();
            }
        }
    }

    mutual_information
}

pub fn calculate_nmi(true_labels: &[usize], predicted_labels: &[usize]) -> f64 {
    let h_true = entropy(true_labels);
    let h_pred = entropy(predicted_labels);
    let mi = calculate_mutual_information(true_labels, predicted_labels);
    2.0 * mi / (h_true + h_pred)
}

pub fn calculate_ari(true_labels: &[usize], predicted_labels: &[usize]) -> f64 {
    let n = true_labels.len();
    let mut contingency_table: HashMap<usize, HashMap<usize, usize>> = HashMap::new();

    for (&true_label, &predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        *contingency_table.entry(predicted_label).or_default().entry(true_label).or_default() += 1;
    }

    let mut sum_combinations = 0;
    let mut sum_row_combinations = 0;
    let mut sum_col_combinations = 0;

    for cluster in contingency_table.values() {
        let row_sum: usize = cluster.values().sum();
        sum_row_combinations += row_sum * (row_sum - 1) / 2;
        for &nij in cluster.values() {
            sum_combinations += nij * (nij - 1) / 2;
        }
    }

    for &label in true_labels {
        let col_sum = predicted_labels.iter().filter(|&&pred_label| pred_label == label).count();
        sum_col_combinations += col_sum * (col_sum - 1) / 2;
    }

    let expected_index = (sum_row_combinations as f64) * (sum_col_combinations as f64) / (n * (n - 1) / 2) as f64;
    let max_index = 0.5 * (sum_row_combinations + sum_col_combinations) as f64;
    (sum_combinations as f64 - expected_index) / (max_index - expected_index)
}

pub fn compute_confusion_matrix(true_labels: &[usize], predicted_labels: &[usize], num_classes: usize) -> DMatrix<usize> {
    let mut confusion_matrix = DMatrix::zeros(num_classes, num_classes);

    for (&true_label, &predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        if true_label < num_classes && predicted_label < num_classes {
            confusion_matrix[(true_label, predicted_label)] += 1;
        }
    }

    confusion_matrix
}

pub fn find_best_permutation_matrix(confusion_matrix: &DMatrix<usize>) -> DMatrix<usize> {
    let n = confusion_matrix.nrows();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut best_perm = perm.clone();
    let mut max_diagonal_sum = 0;

    loop {
        let current_sum: usize = (0..n).map(|i| confusion_matrix[(i, perm[i])]).sum();
        if current_sum > max_diagonal_sum {
            max_diagonal_sum = current_sum;
            best_perm = perm.clone();
        }
        if !next_permutation(&mut perm) {
            break;
        }
    }

    let mut permuted_matrix = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            permuted_matrix[(i, j)] = confusion_matrix[(i, best_perm[j])];
        }
    }

    permuted_matrix
}

pub fn next_permutation<T: Ord>(slice: &mut [T]) -> bool {
    if slice.len() < 2 {
        return false;
    }

    let mut i = slice.len() - 1;
    while i > 0 && slice[i - 1] >= slice[i] {
        i -= 1;
    }

    if i == 0 {
        slice.reverse();
        return false;
    }

    let mut j = slice.len() - 1;
    while slice[j] <= slice[i - 1] {
        j -= 1;
    }

    slice.swap(i - 1, j);
    slice[i..].reverse();
    true
}