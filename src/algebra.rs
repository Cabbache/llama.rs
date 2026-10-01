use std::iter::zip;

use crate::safetensors::Tensor;

pub struct Matrix {
  pub data: Vec<f32>,//row,row,...
  pub shape: (usize, usize)//(num rows, row size)
}

impl Matrix {
  fn new(shape: (usize, usize)) -> Self {
    Matrix {
      data: vec![0.0; shape.0*shape.1],
      shape
    }
  }
}

impl From<Tensor> for Matrix {
  fn from(t: Tensor) -> Self {
    Matrix {
      data: t.values,
      shape: (t.num_rows, t.row_size)
    }
  }
}

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
  //println!("{} {}", mean, variance);
  for x in v.iter_mut() {
		*x = (*x - mean) / (variance + layer_norm_epsilon).powf(0.5)
  }
}

pub fn vector_mean(v1: &Vec<f32>) -> f32 {
	let mut mean: f32 = 0.0;
  //println!("{:?}", v1);
	for (i, v) in v1.iter().enumerate() {
		let i_float = i as f32;
		mean *= i_float;
		mean += v;
		mean /= i_float + 1.0;
    //println!("{}", mean);
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

pub fn matmul(m1: &Matrix, m2: &Matrix) -> Matrix {
  assert_eq!(m1.shape.1, m2.shape.0);
  let mut output = Matrix::new((m1.shape.0, m2.shape.1));
  for row_idx in 0..m1.shape.0 {
    for col_idx in 0..m2.shape.1 {
      let mut total: f32 = 0.0;
      for i in 0..m1.shape.0 {
        let product = m1.data[row_idx*m1.shape.0 + i] * m2.data[i*m2.shape.0 + m2.shape.1];
        total += product;
      }
      output.data[row_idx*m1.shape.0 + col_idx] = total;
    }
  }
  output
}
