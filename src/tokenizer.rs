use core::error::Error;
use serde::Deserialize;
use std::collections::HashMap;
use std::iter::zip;
use std::ops::RangeInclusive;

//https://github.com/openai/gpt-2/blob/master/src/encoder.py?utm_source=chatgpt.com#L9
fn bytes_to_unicode() -> HashMap<u8, char> {
	let ranges: Vec<RangeInclusive<char>> =
		vec!['!'..='~', '¡'..='¬', '®'..='ÿ'];
	let mut n = 0;
	let mut values: Vec<u8> = ranges
		.iter()
		.flat_map(|r| (*r.start() as u8)..(*r.end() as u8))
		.collect();
	let mut copy: Vec<u32> = values.iter().map(|&v| v as u32).collect();
	for b in 0..=255 {
		match values.contains(&b) {
			true => continue,
			false => {
				values.push(b);
				copy.push(256 + n);
				n += 1;
			}
		}
	}
	let c: Vec<char> =
		copy.iter().map(|&c| char::from_u32(c).unwrap()).collect();

	zip(values, c).collect()
}

#[derive(Deserialize, Debug)]
pub struct Model {
	vocab: HashMap<String, usize>,
	merges: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct TokenizerConfig {
	model: Model,
}

#[derive(Debug)]
pub struct Tokenizer {
	config: TokenizerConfig,
	bytelevel_mapping: HashMap<u8, char>,
	inverse_bytelevel_mapping: HashMap<char, u8>,
	mapped_merges: HashMap<String, usize>,
	inverse_vocab: HashMap<usize, String>,
}

impl Tokenizer {
	pub fn from_file(name: String) -> Result<Self, Box<dyn Error>> {
		let json_bytes = std::fs::read(name)?;
		let json_string = String::from_utf8(json_bytes)?;
		let tokenizer: TokenizerConfig =
			serde_json::from_str(&json_string)?;
		let mapped_merges = tokenizer
			.model
			.merges
			.iter()
			.enumerate()
			.map(|(i, s)| (s.clone(), i))
			.collect();

		let mapping = bytes_to_unicode();
		let inverse: HashMap<char, u8> =
			mapping.clone().into_iter().map(|(k, v)| (v, k)).collect();

		let inverse_vocab: HashMap<usize, String> = tokenizer
			.model
			.vocab
			.clone()
			.into_iter()
			.map(|(k, v)| (v, k))
			.collect();

		Ok(Self {
			config: tokenizer,
			bytelevel_mapping: mapping,
			inverse_bytelevel_mapping: inverse,
			mapped_merges,
			inverse_vocab,
		})
	}

	pub fn decode_token(&self, id: usize) -> String {
		let encoded_token =
			self.inverse_vocab.get(&id).expect("expected token");
		let x: Vec<u8> = encoded_token
			.chars()
			.map(|c| {
				*self
					.inverse_bytelevel_mapping
					.get(&c)
					.expect("expected char")
			})
			.collect();
		str::from_utf8(&x).unwrap().to_string()
	}

	pub fn decode_tokens(&self, tokens: &Vec<usize>) -> String {
		let parts: Vec<String> =
			tokens.iter().map(|id| self.decode_token(*id)).collect();
		parts.join("")
	}

	fn apply_mapping(&self, prompt: &String) -> Vec<char> {
		let prompt_bytes: Vec<u8> = prompt.as_bytes().to_vec();
		let mapped_prompt: Vec<char> = prompt_bytes
			.iter()
			.map(|b| *self.bytelevel_mapping.get(b).unwrap())
			.collect();
		mapped_prompt
	}

	fn chars_to_string_tokens(
		&self,
		characters: &Vec<char>,
	) -> Vec<String> {
		let mut result: Vec<String> =
			characters.into_iter().map(|c| c.to_string()).collect();

		while let Some((_, id)) = result
			.windows(2)
			.enumerate()
			.filter_map(|(i, ab)| {
				let k = format!("{} {}", ab[0], ab[1]);
				self.mapped_merges.get(&k).map(|pos| (*pos, i))
			})
			.min_by(|(pos1, _), (post2, _)| pos1.cmp(post2))
		{
			result[id] = result[id].clone() + &result[id + 1];
			result.remove(id + 1);
		}
		result
	}

	fn string_tokens_to_vectors(
		&self,
		tokens: &Vec<String>,
	) -> Vec<usize> {
		tokens
			.into_iter()
			.map(|token| *self.config.model.vocab.get(token).unwrap())
			.collect()
	}

	pub fn tokenize(&self, prompt: &String) -> Vec<usize> {
		let mapped: Vec<char> = self.apply_mapping(prompt);
		let tokens: Vec<String> = self.chars_to_string_tokens(&mapped);
		let vector_tokens: Vec<usize> =
			self.string_tokens_to_vectors(&tokens);
		vector_tokens
	}
}
