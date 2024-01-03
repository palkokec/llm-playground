use orca::llm::quantized::{Model, Quantized};
use orca::pipeline::simple::LLMPipeline;
use orca::pipeline::Pipeline;


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let prompt = "The eiffel tower is";
    let mistral = Quantized::new()
    .with_model(orca::llm::quantized::Model::Mistral7bInstruct)
    .load_model_from_path("../models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")?
    .build_model()?;

    let pipe = LLMPipeline::new(&mistral)
    .load_template("query", prompt)?;
    let result = pipe.execute("query").await?;

    println!("{}", result.content());

    Ok(())
}