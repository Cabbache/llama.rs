use std::iter::zip;

use crate::safetensors::Tensor;

pub fn add_vectors_in_place(v1: &mut Vec<f32>, v2: &[f32]) {
	for (x, y) in zip(v1.iter_mut(), v2.iter()) {
		*x += y
	}
}

pub fn vector_variance(v1: &Vec<f32>) -> f32 {
	let mean = vector_mean(v1);
	let part1: Vec<f32> =
		v1.iter().map(|v| (v - mean).powf(2f32)).collect();
	vector_mean(&part1)
}

pub fn normalize_in_place(v: &mut Vec<f32>, layer_norm_epsilon: f32) {
	let mean = vector_mean(&v);
	let variance = vector_variance(&v);
	for x in v.iter_mut() {
		*x = (*x - mean) / (variance + layer_norm_epsilon).powf(0.5)
	}
}

pub fn vector_mean(v1: &Vec<f32>) -> f32 {
	let mut mean: f32 = 0.0;
	for (i, v) in v1.iter().enumerate() {
		let i_float = i as f32;
		mean *= i_float;
		mean += v;
		mean /= i_float + 1.0;
	}
	mean

	/*
	  * TODO find why this doesnt compile
		let total: Option<f32> = v1
			.into_iter()
			.enumerate()
			.reduce(|acc, (i, v)| (0, (v + acc.1 * (i + 1) as f32) / (i + 2) as f32));
	*/
}

pub fn matmul(m1: &Tensor, m2: &Tensor) -> Tensor {
	assert_eq!(m1.shape.1, m2.shape.0);
	let mut output = Tensor::new((m1.shape.0, m2.shape.1));
	for row_idx in 0..m1.shape.0 {
		for col_idx in 0..m2.shape.1 {
			let mut total: f32 = 0.0;
			for i in 0..m1.shape.0 {
				let product = m1.values[row_idx * m1.shape.0 + i]
					* m2.values[i * m2.shape.0 + col_idx];
				total += product;
			}
			output.values[row_idx * m1.shape.0 + col_idx] = total;
		}
	}
	output
}

pub fn compute_attention(Q: &Tensor, K: &Tensor, V: &Tensor) -> Tensor {
	assert_eq!(Q.shape, K.shape);
	assert_eq!(K.shape, V.shape);
	todo!()
}
