use tokenizers::tokenizer::{Result, Tokenizer};
use models::quantized::{Config, Quantized};

// use orca::llm::quantized::{Model, Quantized};
// use orca::pipeline::simple::LLMPipeline;
// use orca::pipeline::Pipeline;


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let prompt = "The eiffel tower is";
    let weights = std::path::Path::new("../models/mistral-7b-instruct-v0.1.Q4_K_M.gguf");
    let tokenizer = std::path::Path::new("../models/mistral-tokenizer.json");
    let tokenizer = std::fs::read(tokenizer).unwrap();
    let weights = std::fs::read(weights).unwrap();
    let config = Config::default();
    let mistral = Quantized::from_gguf_stream(weights, tokenizer, config).unwrap();
    let mut output = std::io::stdout();
    mistral.generate(prompt, 100, &mut output).unwrap();

    // let model = Quantized::new()
    //     .with_model(Model::Mistral7bInstruct)
    //     .with_sample_len(99)
    //     .load_model_from_path("../models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")?
    //     .build_model()?;

    // let pipe = LLMPipeline::new(&model)
    //     .load_template("greet", "{{#chat}}{{#user}}Hi how are you doing?{{/user}}{{/chat}}")
    //     .unwrap();
    // let result = pipe.execute("greet").await?;

    // println!("{}", result.content());

    Ok(())
}