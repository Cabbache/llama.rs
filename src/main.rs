use crate::algebra::matmul;
use crate::safetensors::SFObject;
use clap::Parser;
use core::error::Error;
use safetensors::SafeTensors;
use safetensors::Tensor;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Read;

use serde::Deserialize;

mod algebra;
mod attention;
mod safetensors;
mod tokenizer;

use crate::attention::attention;
use crate::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
struct Cli {
	#[arg(required = true)]
	safetensors: String,
	#[arg(required = true)]
	tokenizer: String,
	#[arg(required = true)]
	config: String,
	#[arg(short, long)]
	dump: bool,
}

#[derive(Deserialize, Debug)]
struct Config {
	layer_norm_epsilon: f32,
	n_head: usize,
	activation_function: String,
}

const WTE_KEY: &str = "wte.weight";

fn main() -> Result<(), Box<dyn Error>> {
	let args = Cli::parse();
	let mut sf: std::fs::File = fs::File::open(args.safetensors)?;
	let tokenizer: Tokenizer = Tokenizer::from_file(args.tokenizer)?;

	let config_bytes = std::fs::read(args.config)?;
	let config_string: String = String::from_utf8(config_bytes)?;
	let config: Config = serde_json::from_str(&config_string)?;

	let mut buf: [u8; 8] = [0; 8];
	sf.read_exact(&mut buf)?;
	let jsonlen = u64::from_le_bytes(buf);
	let mut newbuf: Vec<u8> = vec![0; jsonlen as usize];
	sf.read_exact(&mut newbuf)?;
	let jsonstring = String::from_utf8(newbuf).expect("fuck");
	if args.dump {
		println!("{}", jsonstring);
		return Ok(());
	}
	let smeta: SFObject = serde_json::from_str(&jsonstring)?;
	let sf_file: SafeTensors = SafeTensors {
		jsonlen,
		header: smeta,
	};

	print!("> ");
	let _ = io::Write::flush(&mut io::stdout());

	let mut prompt = String::new();
	let stdin = io::stdin();
	let mut handle = stdin.lock();
	handle.read_line(&mut prompt)?;

	let tokens = tokenizer.tokenize(&prompt);
	println!("{:?}", tokens);

	let wte_tensor = sf_file.load_tensor(WTE_KEY, &sf);

	let mut embeddings: Tensor = tokens
		.iter()
		.enumerate()
		.map(|(index, token)| {
			let mut row = wte_tensor.getrow(index);
			let token_embedding = wte_tensor.getrow(*token);
			row.elementwise_add(&token_embedding);
			row
		})
		.reduce(|mut acc, v| {
			acc.concat_rows(&v);
			acc
		})
		.expect("???");

	//keep copy because we need it later unchanged
	let input_block = embeddings.clone();

	for i in 0..=10 {
		embeddings = attention(embeddings, &config, &sf_file, &sf, 0);
		println!("{}", i);
	}

	Ok(())
}
