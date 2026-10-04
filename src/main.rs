use crate::algebra::matmul;
use crate::algebra::normalize_in_place;
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
	#[arg(required = true)]
	config: String,
	#[arg(short, long)]
	dump: bool,
}

#[derive(Deserialize, Debug)]
struct Config {
	layer_norm_epsilon: f32,
	n_head: usize,
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

	let mut embeddings: Vec<Vec<f32>> = tokens
		.iter()
		.enumerate()
		.map(|(index, token)| {
			let mut row = wte_tensor.getrow(index).to_vec();
			let token_embedding = wte_tensor.getrow(*token);
			add_vectors_in_place(&mut row, token_embedding);
			row
		})
		.collect();

	let h0_bias_tensor = sf_file.load_tensor("h.0.ln_1.bias", &sf);
	let h0_weight_tensor = sf_file.load_tensor("h.0.ln_1.weight", &sf);
	let h0_bias = h0_bias_tensor.getrow(0);
	let h0_weight = h0_weight_tensor.getrow(0);

	for mut row in &mut embeddings {
		normalize_in_place(row, config.layer_norm_epsilon);
		for i in 0..row.len() {
			row[i] = row[i] * h0_weight[i] + h0_bias[i];
		}
	}
	let mut flattened_embeddings: Vec<f32> = Vec::new();
	for row in embeddings.iter_mut() {
		flattened_embeddings.append(&mut *row);
	}

	let embedding_matrix = Tensor {
		values: flattened_embeddings,
		shape: (tokens.len(), wte_tensor.shape.1),
	};

	let h0_attn_weight_tensor =
		sf_file.load_tensor("h.0.attn.c_attn.weight", &sf);
	let h0_attn_bias_tensor =
		sf_file.load_tensor("h.0.attn.c_attn.bias", &sf);

	let mut result = matmul(&embedding_matrix, &h0_attn_weight_tensor);
	result.add_bias(&h0_attn_bias_tensor);

	let mut heads = result.get_heads(config.n_head);
	for mut head in heads {
		head.key.transpose_mut();
		let mut attention = matmul(&head.query, &head.key);
		attention.scalar_multiply(1.0 / (64f32).sqrt());
		attention.mask_diagonal();
		println!("{:?}", attention);
	}

	Ok(())
}
