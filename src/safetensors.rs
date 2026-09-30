use serde::Deserialize;
use std::io;
use std::io::Seek;
use std::io::Read;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct SFObject(HashMap<String, SFMeta>);

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
	values: Vec<f32>,
	num_rows: usize,
	row_size: usize,
}

impl SFObject {
	pub fn get_tensor_info(&self, key: &str) -> &TensorInfo {
		let result = match self.0.get(key) {
			Some(SFMeta::Tensor(t)) => t,
			_ => panic!("unexpected error when accessing key {}", key),
		};
    assert_eq!(result.shape.len(), 2);
    assert!(matches!(result.dtype, WeightType::F32));
    result
	}
}

impl Tensor {
	pub fn getrow(&self, idx: usize) -> &[f32] {
		&self.values[self.row_size * idx..self.row_size * (idx + 1)]
	}
}

impl TensorInfo {
	pub fn load_tensor(&self, mut sf: &std::fs::File) -> Tensor {
		let start = self.data_offsets[0];
		let end = self.data_offsets[1];
		assert_eq!(self.shape[0] * self.shape[1] * 4, end - start);
		sf.seek(io::SeekFrom::Start(start as u64));
		let mut raw_bytes: Vec<u8> = vec![0; end - start];
		sf.read_exact(&mut raw_bytes);
		let values = raw_bytes
			.chunks_exact(4)
			.map(TryInto::try_into)
			.map(Result::unwrap)
			.map(f32::from_le_bytes)
			.collect();
		Tensor {
			values: values,
			num_rows: self.shape[0],
			row_size: self.shape[1],
		}
	}
}
