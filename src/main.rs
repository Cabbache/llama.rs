use clap::Parser;
use core::error::Error;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Read;
use std::io::Seek;
use std::iter::zip;

use crate::tokenizer::Tokenizer;

mod tokenizer;

#[derive(Parser, Debug)]
struct Cli {
	#[arg(required = true)]
	safetensors: String,
	#[arg(required = true)]
	tokenizer: String,
	#[arg(short, long)]
	dump: bool,
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
	values: Vec<f32>,
	num_rows: usize,
	row_size: usize,
}

fn add_vectors_in_place(v1: &mut Vec<f32>, v2: &[f32]) {
	for (mut x, y) in zip(v1.iter_mut(), v2.iter()) {
		*x += y
	}
}

impl Tensor {
	fn getrow(&self, idx: usize) -> &[f32] {
		&self.values[self.row_size * idx..self.row_size * (idx + 1)]
	}
}

impl TensorInfo {
	fn load_tensor(&self, mut sf: &std::fs::File) -> Tensor {
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

#[derive(Deserialize, Debug)]
pub struct SFObject(HashMap<String, SFMeta>);

impl SFObject {
	pub fn get_tensor_info(&self, key: &str) -> &TensorInfo {
		match self.0.get(key) {
			Some(SFMeta::Tensor(t)) => t,
			_ => panic!("unexpected error when accessing key {}", key),
		}
	}
}

const WTE_KEY: &str = "wte.weight";

fn main() -> Result<(), Box<dyn Error>> {
	let args = Cli::parse();
	let mut sf: std::fs::File = fs::File::open(args.safetensors)?;
	let tokenizer: Tokenizer = Tokenizer::from_file(args.tokenizer)?;

	let mut buf: [u8; 8] = [0; 8];
	sf.read_exact(&mut buf)?;
	let jsonlen = usize::from_le_bytes(buf);
	let mut newbuf: Vec<u8> = vec![0; jsonlen];
	sf.read_exact(&mut newbuf)?;
	let jsonstring = String::from_utf8(newbuf).expect("fuck");
	if args.dump {
		println!("{}", jsonstring);
		return Ok(());
	}
	let smeta: SFObject = serde_json::from_str(&jsonstring)?;

	print!("> ");
	io::Write::flush(&mut io::stdout());

	let mut prompt = String::new();
	let stdin = io::stdin();
	let mut handle = stdin.lock();
	handle.read_line(&mut prompt)?;

	let tokens = tokenizer.tokenize(&prompt);
	println!("{:?}", tokens);

	let wte = smeta.get_tensor_info(WTE_KEY);

	assert_eq!(wte.shape.len(), 2);
	assert!(matches!(wte.dtype, WeightType::F32));

	let wte_tensor = wte.load_tensor(&sf);

	let embeddings: Vec<Vec<f32>> = tokens
		.iter()
		.enumerate()
		.map(|(index, token)| {
			let mut row = wte_tensor.getrow(index).clone().to_vec();
			let token_embedding = wte_tensor.getrow(*token);
			add_vectors_in_place(&mut row, token_embedding);
			row
		})
		.collect();

	println!("{:?}", embeddings);

	Ok(())
}
