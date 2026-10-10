use core::f32::consts::{self, PI};

use crate::safetensors::Tensor;

pub fn slice_variance(v1: &[f32]) -> f32 {
	let mean = slice_mean(v1);
	let part1: Vec<f32> =
		v1.iter().map(|v| (v - mean).powf(2f32)).collect();
	slice_mean(&part1)
}

pub fn slice_mean(v1: &[f32]) -> f32 {
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
			for i in 0..m1.shape.1 {
				let product = m1.values[row_idx * m1.shape.1 + i]
					* m2.values[i * m2.shape.1 + col_idx];
				total += product;
			}
			output.values[row_idx * m2.shape.1 + col_idx] = total;
		}
	}
	output
}

pub fn tanh(x: f32) -> f32 {
	if x > 5.0 {
		return 1.0;
	} else if x < -5.0 {
		return -1.0;
	} else {
		let e = consts::E;
		let pow2x = e.powf(2.0 * x);
		let out = (pow2x - 1.0) / (pow2x + 1.0);
		out
	}
}

pub fn gelu_new(x: f32) -> f32 {
	let twoonpisqrt = (2.0 / PI).sqrt();
	let tanh_input = (x + x.powf(3f32) * 0.044715) * twoonpisqrt;
	let oneplus = 1.0 + tanh(tanh_input);
	let out = 0.5 * x * oneplus;
	out
}
