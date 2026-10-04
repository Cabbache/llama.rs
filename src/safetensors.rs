use serde::Deserialize;
use std::collections::HashMap;
use std::io;
use std::io::Read;
use std::io::Seek;

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

#[derive(Debug)]
pub struct Tensor {
	pub values: Vec<f32>,
	pub shape: (usize, usize),
}

#[derive(Debug)]
pub struct Head {
	pub Q: Tensor,
	pub K: Tensor,
	pub V: Tensor,
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

	pub fn add_bias(&mut self, bias: &Tensor) {
		assert_eq!(bias.shape.0, 1);
		assert_eq!(self.shape.1, bias.shape.1);
		for i in 0..self.shape.0 {
			for j in 0..self.shape.1 {
				self.values[i * self.shape.1 + j] += bias.values[j];
			}
		}
	}

	pub fn get_heads(&self) -> Vec<Head> {
		let mut heads: Vec<Head> = Vec::new();
		let Q = self.slice((0, 0), (self.shape.0, 768));
		let K = self.slice((0, 768), (self.shape.0, 768 * 2));
		let V = self.slice((0, 768 * 2), (self.shape.0, 768 * 3));

		for i in 0..12 {
			let sl_p1 = (0, i * 64);
			let sl_p2 = (self.shape.0, (i + 1) * 64);
			heads.push(Head {
				Q: Q.slice(sl_p1, sl_p2),
				K: K.slice(sl_p1, sl_p2),
				V: V.slice(sl_p1, sl_p2),
			});
		}
		heads
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

		assert!(min_x >= 0);
		assert!(min_y >= 0);
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

	pub fn getrow(&self, idx: usize) -> &[f32] {
		&self.values[self.shape.1 * idx..self.shape.1 * (idx + 1)]
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
			//.map(f32::from_be_bytes)
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
