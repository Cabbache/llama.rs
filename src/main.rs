use clap::Parser;
use core::error::Error;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Read;

use crate::tokenizer::Tokenizer;

mod tokenizer;

#[derive(Parser, Debug)]
struct Cli {
	#[arg(required = true)]
	safetensors: String,
	#[arg(required = true)]
	tokenizer: String,
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
	let smeta: HashMap<String, SFMeta> =
		serde_json::from_str(&jsonstring)?;

	print!("> ");
	io::Write::flush(&mut io::stdout());

	let mut prompt = String::new();
	let stdin = io::stdin();
	let mut handle = stdin.lock();
	handle.read_line(&mut prompt)?;

	println!("{:?}", tokenizer.tokenize(&prompt));
	Ok(())
}
