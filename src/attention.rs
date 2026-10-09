use crate::algebra::gelu_new;
use crate::safetensors::Tensor;
use crate::{Config, matmul, safetensors::SafeTensors};

fn layername(block_number: usize, sname: &str) -> String {
	format!("h.{}.{}", block_number, sname)
}

pub fn attention(
	mut input: Tensor,
	config: &Config,
	sf_file: &SafeTensors,
	sf: &std::fs::File,
	block_number: usize,
) -> Tensor {
	let input_block = input.clone();

	//layernorm 1
	let h0_weight = sf_file
		.load_tensor(&layername(block_number, "ln_1.weight"), &sf);
	let h0_bias =
		sf_file.load_tensor(&layername(block_number, "ln_1.bias"), &sf);
	input.normalize(config.layer_norm_epsilon);
	input.rowwise_mul(&h0_weight);
	input.rowwise_add(&h0_bias);

	let h0_attn_weight_tensor = sf_file.load_tensor(
		&layername(block_number, "attn.c_attn.weight"),
		&sf,
	);
	let h0_attn_bias_tensor = sf_file
		.load_tensor(&layername(block_number, "attn.c_attn.bias"), &sf);

	let mut result = matmul(&input, &h0_attn_weight_tensor);
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

	let h0_proj_weight = sf_file.load_tensor(
		&layername(block_number, "attn.c_proj.weight"),
		&sf,
	);
	let h0_proj_bias = sf_file
		.load_tensor(&layername(block_number, "attn.c_proj.bias"), &sf);
	head_output.matmul_inplace(&h0_proj_weight);
	head_output.rowwise_add(&h0_proj_bias);

	let before_residual_1 = head_output.clone();

	//residual connection 1
	head_output.elementwise_add(&input_block);

	//layernorm 2
	let h0_weight = sf_file
		.load_tensor(&layername(block_number, "ln_2.weight"), &sf);
	let h0_bias =
		sf_file.load_tensor(&layername(block_number, "ln_2.bias"), &sf);
	head_output.normalize(config.layer_norm_epsilon);
	head_output.rowwise_mul(&h0_weight);
	head_output.rowwise_add(&h0_bias);

	//mlp fc
	let mlp_weight_fc = sf_file
		.load_tensor(&layername(block_number, "mlp.c_fc.weight"), &sf);
	let mlp_bias_fc = sf_file
		.load_tensor(&layername(block_number, "mlp.c_fc.bias"), &sf);
	let mut mlp_output = matmul(&head_output, &mlp_weight_fc);
	mlp_output.rowwise_add(&mlp_bias_fc);

	//gelu
	assert_eq!(config.activation_function, "gelu_new");
	mlp_output.values.iter_mut().for_each(|v| *v = gelu_new(*v));

	//mlp
	let mlp_weight = sf_file.load_tensor(
		&layername(block_number, "mlp.c_proj.weight"),
		&sf,
	);
	let mlp_bias = sf_file
		.load_tensor(&layername(block_number, "mlp.c_proj.bias"), &sf);
	let mut mlp_output = matmul(&mlp_output, &mlp_weight);
	mlp_output.rowwise_add(&mlp_bias);

	//residual connection 2
	mlp_output.elementwise_add(&before_residual_1);

	mlp_output
}
