use crate::algebra::matmul;
use crate::attention::layernorm;
use crate::safetensors::SFObject;
use clap::Parser;
use core::error::Error;
use safetensors::SafeTensors;
use safetensors::Tensor;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Read;
use std::io::Write;

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
	#[arg(long)]
	disable_cache: bool,
}

#[derive(Deserialize, Debug)]
struct Config {
	layer_norm_epsilon: f32,
	n_head: usize,
	activation_function: String,
}

const WTE_KEY: &str = "wte.weight";
const WPE_KEY: &str = "wpe.weight";

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
	let mut sf_file =
		SafeTensors::new(jsonlen, smeta, args.disable_cache);

	print!("> ");
	let _ = io::Write::flush(&mut io::stdout());

	let mut prompt = String::new();
	let stdin = io::stdin();
	let mut handle = stdin.lock();
	handle.read_line(&mut prompt)?;
	prompt = prompt.trim().to_string();
	println!("'{}'", prompt);

	let mut tokens = tokenizer.tokenize(&prompt);
	println!("tokens: {:?}", tokens);
	loop {
		let mut wte_tensor = sf_file.load_tensor(WTE_KEY, &sf);
		let wpe_tensor = sf_file.load_tensor(WPE_KEY, &sf);

		let mut embeddings: Tensor = tokens
			.iter()
			.enumerate()
			.map(|(position, token_id)| {
				let mut row = wte_tensor.getrow(*token_id);
				let token_embedding = wpe_tensor.getrow(position);
				row.elementwise_add(&token_embedding);
				row
			})
			.reduce(|mut acc, v| {
				acc.concat_rows(&v);
				acc
			})
			.expect("???");

		//process all attention blocks
		let num_blocks = 12;
		for i in 0..num_blocks {
			let bar = "=".repeat(i) + ">" + &" ".repeat(num_blocks - i);
			print!("\r[{}] [{}/{}]", bar, i + 1, num_blocks);
			let _ = io::stdout().flush();
			embeddings =
				attention(embeddings, &config, &mut sf_file, &sf, i);
		}
		println!();

		let ln_f_weight = sf_file.load_tensor("ln_f.weight", &sf);
		let ln_f_bias = sf_file.load_tensor("ln_f.bias", &sf);
		layernorm(
			&mut embeddings,
			&ln_f_weight,
			&ln_f_bias,
			config.layer_norm_epsilon,
		);

		let lastrow = embeddings.getrow(embeddings.shape.0 - 1);
		wte_tensor.transpose_mut();
		let logits = matmul(&lastrow, &wte_tensor);
		//println!("{:?}", logits);
		let (next_token_id, _) = logits
			.values
			.iter()
			.enumerate()
			.max_by(|(_, a), (_, b)| {
				a.partial_cmp(b).expect("f32 issue")
			})
			.expect("no tokens");

		tokens.push(next_token_id);
		println!("output: '{}'", tokenizer.decode_tokens(&tokens));
	}
}
