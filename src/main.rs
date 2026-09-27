use core::{error::Error};
use std::fs;
use std::io::Read;
use clap::Parser;
use serde_json::Value;

#[derive(Parser, Debug)]
struct Cli {
  #[arg(required = true)]
  safetensors: String,
}

fn main() -> Result<(), Box<dyn Error>>{
  let args = Cli::parse();
  let mut sf: std::fs::File = fs::File::open(args.safetensors)?;
  let mut buf: [u8; 8] = [0; 8];
  sf.read_exact(&mut buf)?;
  let jsonlen = usize::from_le_bytes(buf);
  eprintln!("len = {}", jsonlen);
  let mut newbuf: Vec<u8> = vec![0; jsonlen];
  sf.read_exact(&mut newbuf)?;
  let jsonstring = String::from_utf8(newbuf).expect("fuck");
  println!("{}", jsonstring);
  let smeta: Value = serde_json::from_str(&jsonstring)?;
  //println!("{:?}", smeta);
  Ok(())
}
