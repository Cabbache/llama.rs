use std::iter::zip;

pub fn add_vectors_in_place(v1: &mut Vec<f32>, v2: &[f32]) {
	for (x, y) in zip(v1.iter_mut(), v2.iter()) {
		*x += y
	}
}

pub fn vector_variance(v1: &Vec<f32>) -> f32 {
  let mean = vector_mean(v1);
  let part1: Vec<f32> = v1.iter().map(|v| (v-mean).powf(2f32)).collect();
  vector_mean(&part1)
}

pub fn normalize_in_place(v: &mut Vec<f32>, layer_norm_epsilon: f32) {
  let mean = vector_mean(&v);
  let variance = vector_variance(&v);
  v.iter_mut().map(|&mut v| (v-mean)/(variance+layer_norm_epsilon).powf(0.5));
}

pub fn vector_mean(v1: &Vec<f32>) -> f32 {
  let mut mean: f32 = 0.0;
  for (i,v) in v1.iter().enumerate() {
    let i_float = i as f32;
    mean *= i_float;
    mean += v;
    mean /= i_float+1.0;
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
