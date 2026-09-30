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

#[derive(Deserialize)]
pub struct Model {
	vocab: HashMap<String, usize>,
	merges: Vec<String>,
}

#[derive(Deserialize)]
pub struct TokenizerConfig {
	model: Model,
}

pub struct Tokenizer {
	config: TokenizerConfig,
	bytelevel_mapping: HashMap<u8, char>,
	mapped_merges: HashMap<String, usize>,
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

		Ok(Self {
			config: tokenizer,
			bytelevel_mapping: bytes_to_unicode(),
			mapped_merges,
		})
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
		//println!("{:?}", tokens);
		let vector_tokens: Vec<usize> =
			self.string_tokens_to_vectors(&tokens);
		vector_tokens
	}
}
