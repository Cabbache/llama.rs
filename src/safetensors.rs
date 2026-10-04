use core::f32;
use serde::Deserialize;
use std::collections::HashMap;
use std::io;
use std::io::Read;
use std::io::Seek;
use std::iter::zip;

use crate::algebra::slice_mean;
use crate::algebra::slice_variance;

use crate::matmul;

#[derive(Deserialize, Debug)]
pub struct SFObject(HashMap<String, SFMeta>);

#[derive(Debug)]
pub struct SafeTensors {
	pub jsonlen: u64,
	pub header: SFObject,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum SFMeta {
	Tensor(TensorInfo),
	Metadata(MetaFormat),
}

#[derive(Deserialize, Debug)]
pub struct MetaFormat {
	format: String,
}

#[derive(Deserialize, Debug)]
pub enum WeightType {
	F32,
}

#[derive(Deserialize, Debug)]
pub struct TensorInfo {
	dtype: WeightType,
	shape: Vec<usize>,
	data_offsets: [usize; 2],
}

#[derive(Debug, Clone)]
pub struct Tensor {
	pub values: Vec<f32>,
	pub shape: (usize, usize),
}

#[derive(Debug)]
pub struct Head {
	pub query: Tensor,
	pub key: Tensor,
	pub value: Tensor,
}

impl SafeTensors {
	pub fn get_tensor_info(&self, key: &str) -> &TensorInfo {
		let result = match self.header.0.get(key) {
			Some(SFMeta::Tensor(t)) => t,
			_ => panic!("unexpected error when accessing key {}", key),
		};
		assert!(matches!(result.dtype, WeightType::F32));
		result
	}

	pub fn load_tensor(&self, key: &str, sf: &std::fs::File) -> Tensor {
		let info = self.get_tensor_info(key);
		info.load_tensor(sf, self.jsonlen + 8)
	}
}

impl Tensor {
	pub fn new(shape: (usize, usize)) -> Self {
		Self {
			values: vec![0.0; shape.0 * shape.1],
			shape,
		}
	}

	fn rowwise_update<F>(&mut self, other: &Tensor, op: F)
	where
		F: Fn(f32, f32) -> f32,
	{
		assert_eq!(other.shape.0, 1);
		assert_eq!(self.shape.1, other.shape.1);

		for i in 0..self.shape.0 {
			for j in 0..self.shape.1 {
				let idx = i * self.shape.1 + j;
				self.values[idx] =
					op(self.values[idx], other.values[j]);
			}
		}
	}

	pub fn rowwise_add(&mut self, bias: &Tensor) {
		self.rowwise_update(bias, |a, b| a + b);
	}

	pub fn rowwise_mul(&mut self, bias: &Tensor) {
		self.rowwise_update(bias, |a, b| a * b);
	}

	pub fn get_heads(&self, n_head: usize) -> Vec<Head> {
		let mut heads: Vec<Head> = Vec::new();
		let query = self.slice((0, 0), (self.shape.0, 768));
		let key = self.slice((0, 768), (self.shape.0, 768 * 2));
		let value = self.slice((0, 768 * 2), (self.shape.0, 768 * 3));

		for i in 0..n_head {
			let sl_p1 = (0, i * 64);
			let sl_p2 = (self.shape.0, (i + 1) * 64);
			heads.push(Head {
				query: query.slice(sl_p1, sl_p2),
				key: key.slice(sl_p1, sl_p2),
				value: value.slice(sl_p1, sl_p2),
			});
		}
		heads
	}

	pub fn elementwise_add(&mut self, t: &Tensor) {
		assert_eq!(self.shape, t.shape);
		zip(self.values.iter_mut(), t.values.iter())
			.for_each(|(v, c)| *v += c);
	}

	pub fn slice(
		&self,
		p1: (usize, usize),
		p2: (usize, usize),
	) -> Tensor {
		let min_x = p1.0.min(p2.0);
		let max_x = p1.0.max(p2.0);

		let min_y = p1.1.min(p2.1);
		let max_y = p1.1.max(p2.1);

		assert!(max_x <= self.shape.0);
		assert!(max_y <= self.shape.1);

		let new_shape = (max_x - min_x, max_y - min_y);
		assert!(new_shape.0 > 0);
		assert!(new_shape.1 > 0);

		let mut new_data =
			Vec::with_capacity(new_shape.0 * new_shape.1);

		for i in 0..new_shape.0 {
			let base = (i + min_x) * self.shape.1 + min_y;
			let slice = self.values[base..base + new_shape.1].to_vec();
			new_data.extend(slice);
		}

		Tensor {
			values: new_data,
			shape: new_shape,
		}
	}

	pub fn matmul_inplace(&mut self, t: &Tensor) {
		let result = matmul(self, t);
		self.values = result.values;
		self.shape = result.shape;
	}

	pub fn transpose_mut(&mut self) {
		let tmp = self.values.clone();
		let mut ctr = 0;
		for j in 0..self.shape.1 {
			for i in 0..self.shape.0 {
				self.values[ctr] = tmp[i * self.shape.1 + j];
				ctr += 1;
			}
		}
		let (a, b) = self.shape;
		self.shape = (b, a);
	}

	pub fn concat_columns(&mut self, t: &Tensor) {
		assert_eq!(self.shape.0, t.shape.0);
		let new_shape = (self.shape.0, self.shape.1 + t.shape.1);
		let tmp = self.values.clone();
		self.values = vec![0.0; self.area() + t.area()];
		for i in 0..self.shape.0 {
			for j in 0..self.shape.1 {
				self.values[i * new_shape.1 + j] =
					tmp[i * self.shape.1 + j];
			}

			for j in 0..t.shape.1 {
				self.values[i * new_shape.1 + self.shape.1 + j] =
					t.values[i * t.shape.1 + j];
			}
		}
		self.shape = new_shape;
	}

	pub fn concat_rows(&mut self, t: &Tensor) {
		assert_eq!(self.shape.1, t.shape.1);
		self.shape = (self.shape.0 + t.shape.0, self.shape.1);
		self.values.extend_from_slice(&t.values);
	}

	pub fn normalize(&mut self, layer_norm_epsilon: f32) {
		for i in 0..self.shape.0 {
			let slice = self.getrow(i);
			let mean = slice_mean(&slice.values);
			let variance = slice_variance(&slice.values);
			for j in 0..self.shape.1 {
				let x = self.values[i * self.shape.1 + j];
				self.values[i * self.shape.1 + j] = (x - mean)
					/ (variance + layer_norm_epsilon).powf(0.5);
			}
		}
	}

	pub fn scalar_multiply(&mut self, scalar: f32) {
		self.values.iter_mut().for_each(|v| *v *= scalar);
	}

	//used to stop future tokens from relating to previous tokens.
	pub fn mask_diagonal(&mut self) {
		assert_eq!(self.shape.0, self.shape.1);
		for i in 0..self.shape.0 {
			for j in (i + 1)..self.shape.1 {
				self.values[i * self.shape.1 + j] = f32::NEG_INFINITY;
			}
		}
	}

	pub fn getrow(&self, idx: usize) -> Tensor {
		self.slice((idx, 0), (idx + 1, self.shape.1))
	}

	pub fn softmax(&mut self) {
		for i in 0..self.shape.0 {
			let slice = &mut self.values
				[i * self.shape.1..(i + 1) * self.shape.1];
			let mapped: Vec<f32> =
				slice.iter().map(|v| f32::consts::E.powf(*v)).collect();
			let total: f32 = mapped.iter().sum();
			let mapped: Vec<f32> =
				mapped.iter().map(|v| v / total).collect();
			slice.copy_from_slice(&mapped);
		}
	}

	pub fn area(&self) -> usize {
		self.shape.0 * self.shape.1
	}
}

impl TensorInfo {
	fn load_tensor(
		&self,
		mut sf: &std::fs::File,
		offset: u64,
	) -> Tensor {
		let start = self.data_offsets[0];
		let end = self.data_offsets[1];
		let product: usize = self
			.shape
			.iter()
			.cloned()
			.reduce(|acc, v| acc * v)
			.expect("empty shape");
		assert_eq!(product * 4, end - start);

		let _ = sf.seek(io::SeekFrom::Start(offset + start as u64));
		let mut raw_bytes: Vec<u8> = vec![0; end - start];
		let _ = sf.read_exact(&mut raw_bytes);
		let values = raw_bytes
			.chunks_exact(4)
			.map(TryInto::try_into)
			.map(Result::unwrap)
			.map(f32::from_le_bytes)
			.collect();
		Tensor {
			values: values,
			shape: self.get_shape(),
		}
	}

	fn get_shape(&self) -> (usize, usize) {
		match self.shape.len() {
			2 => {
				let (a, b) = (self.shape[0], self.shape[1]);
				(a, b)
			}
			1 => (1, self.shape[0]),
			_ => panic!("unexpected shape"),
		}
	}
}
