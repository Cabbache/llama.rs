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
mod safetensors;
mod tokenizer;

use algebra::gelu_new;

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

	//layernorm 1
	let h0_weight = sf_file.load_tensor("h.0.ln_1.weight", &sf);
	let h0_bias = sf_file.load_tensor("h.0.ln_1.bias", &sf);
	embeddings.normalize(config.layer_norm_epsilon);
	embeddings.rowwise_mul(&h0_weight);
	embeddings.rowwise_add(&h0_bias);

	let h0_attn_weight_tensor =
		sf_file.load_tensor("h.0.attn.c_attn.weight", &sf);
	let h0_attn_bias_tensor =
		sf_file.load_tensor("h.0.attn.c_attn.bias", &sf);

	let mut result = matmul(&embeddings, &h0_attn_weight_tensor);
	result.rowwise_add(&h0_attn_bias_tensor);

	let heads = result.get_heads(config.n_head);
	let mut maybe_head_output: Option<Tensor> = None;
	for mut head in heads {
		head.key.transpose_mut();
		let mut attention = matmul(&head.query, &head.key);
		attention.scalar_multiply(1.0 / (64f32).sqrt());
		attention.mask_diagonal();
		attention.softmax();
		let attention = matmul(&attention, &head.value);
		match maybe_head_output {
			Some(ref mut t) => t.concat_columns(&attention),
			None => maybe_head_output = Some(attention),
		}
	}
	let mut head_output = maybe_head_output.expect("???");

	let h0_proj_weight =
		sf_file.load_tensor("h.0.attn.c_proj.weight", &sf);
	let h0_proj_bias = sf_file.load_tensor("h.0.attn.c_proj.bias", &sf);
	head_output.matmul_inplace(&h0_proj_weight);
	head_output.rowwise_add(&h0_proj_bias);

	let before_residual_1 = head_output.clone();

	//residual connection 1
	head_output.elementwise_add(&input_block);

	//layernorm 2
	let h0_weight = sf_file.load_tensor("h.0.ln_2.weight", &sf);
	let h0_bias = sf_file.load_tensor("h.0.ln_2.bias", &sf);
	head_output.normalize(config.layer_norm_epsilon);
	head_output.rowwise_mul(&h0_weight);
	head_output.rowwise_add(&h0_bias);

	//mlp fc
	let mlp_weight_fc = sf_file.load_tensor("h.0.mlp.c_fc.weight", &sf);
	let mlp_bias_fc = sf_file.load_tensor("h.0.mlp.c_fc.bias", &sf);
	let mut mlp_output = matmul(&head_output, &mlp_weight_fc);
	mlp_output.rowwise_add(&mlp_bias_fc);

	//gelu
	assert_eq!(config.activation_function, "gelu_new");
	mlp_output.values.iter_mut().for_each(|v| *v = gelu_new(*v));

	//mlp
	let mlp_weight = sf_file.load_tensor("h.0.mlp.c_proj.weight", &sf);
	let mlp_bias = sf_file.load_tensor("h.0.mlp.c_proj.bias", &sf);
	let mut mlp_output = matmul(&mlp_output, &mlp_weight);
	mlp_output.rowwise_add(&mlp_bias);

	//residual connection 2
	mlp_output.elementwise_add(&before_residual_1);

	println!("{:?}", mlp_output);

	Ok(())
}
