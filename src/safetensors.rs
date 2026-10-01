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
	pub num_rows: usize,
	pub row_size: usize,
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
	pub fn getrow(&self, idx: usize) -> &[f32] {
		&self.values[self.row_size * idx..self.row_size * (idx + 1)]
	}
}

impl TensorInfo {
	fn load_tensor(&self, mut sf: &std::fs::File, offset: u64) -> Tensor {
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
			num_rows: self.get_shape()[0],
			row_size: self.get_shape()[1],
		}
	}

	fn get_shape(&self) -> [usize; 2] {
		match self.shape.len() {
			2 => {
				let (a, b) = (self.shape[0], self.shape[1]);
				[a, b]
			}
			1 => [1, self.shape[0]],
			_ => panic!("unexpected shape"),
		}
	}
}
