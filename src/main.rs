use crate::safetensors::SFObject;
use clap::Parser;
use core::error::Error;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Read;

mod algebra;
mod safetensors;
mod tokenizer;

use algebra::add_vectors_in_place;
use algebra::vector_mean;

use crate::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
struct Cli {
	#[arg(required = true)]
	safetensors: String,
	#[arg(required = true)]
	tokenizer: String,
	#[arg(short, long)]
	dump: bool,
}

const WTE_KEY: &str = "wte.weight";

fn main() -> Result<(), Box<dyn Error>> {
	let args = Cli::parse();
	let sf: std::fs::File = fs::File::open(args.safetensors)?;
	let tokenizer: Tokenizer = Tokenizer::from_file(args.tokenizer)?;

  return Ok(());

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
	let _ = io::Write::flush(&mut io::stdout());

	let mut prompt = String::new();
	let stdin = io::stdin();
	let mut handle = stdin.lock();
	handle.read_line(&mut prompt)?;

	let tokens = tokenizer.tokenize(&prompt);
	println!("{:?}", tokens);

	let wte_tensor = smeta.load_tensor(WTE_KEY, &sf);

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

	let h0_bias = smeta.load_tensor("h.0.ln_1.bias", &sf);
	let h0_weight = smeta.load_tensor("h.0.ln_1.weight", &sf);

	println!("{:?}", embeddings);
	Ok(())
}
