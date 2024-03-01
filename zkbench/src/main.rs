use std::fs;
use serde::{Serialize, Deserialize};
use serde_json::{Result,Deserializer};
use zkbench::{MsmBenchmark,NttBenchmark,git_id};
use sqlx::postgres::PgPool;
use std::env;
use dotenv::dotenv;

#[derive(Debug, Deserialize)]
pub struct Typical {
    #[serde(rename = "estimate")]
    pub runtime_estimate: f32,
    #[serde(rename = "unit")]
    pub time_unit: String,
}

fn time_unit_to_number(time_str: &str) -> f32 {
    match time_str {
        "ns" => 1e-9,
        "us" => 1e-6,
        _ => 0.0, 
    }
}

fn normalize_runtime(t: Typical ) -> f32 {
    t.runtime_estimate * time_unit_to_number(&t.time_unit)
}

#[derive(Debug, Deserialize)]
pub struct Performance {
    #[serde(rename = "id")]
    pub benchmark_id: String,
    #[serde(rename = "typical")]
    pub typical: Typical
}

#[tokio::main]
async fn main() {
    // export RUST_LOG=sqlx=trace
    env_logger::init();
    dotenv().ok();

    let repository_path = env::var("BENCHMARK_REPO").expect("BENCHMARK_REPO must be set");
    println!("Repository path: {}", repository_path);
    let id = zkbench::git_id(&repository_path);
    println!("Current commit hash: {}", id);


    // experiment with database INSERT
    let path= "../benchmarks/rust/msm";
    let metadata_path = format!("{}/metadata.json", path);
    let metadata_json = fs::read_to_string(metadata_path).expect("Failed to read the file");
    let mut b: MsmBenchmark = serde_json::from_str(&metadata_json.to_string()).expect("Deserialization failed");
    println!("Metadata {:?}",b);
    let performance_path = format!("{}/criterion.json", path);
    let performance_json = fs::read_to_string(performance_path).expect("Failed to read the file");
    // Deserialize the first JSON object in the file
    let mut deserializer = Deserializer::from_str(&performance_json);
    let result: Result<Performance> = Deserialize::deserialize(&mut deserializer);
    let data = result.unwrap();
    println!("Performance: {:?}", data);
    b.runtime_sec = Some(normalize_runtime(data.typical));
    println!("Benchmark: {:?}", b);
    let connect_result = PgPool::connect("postgres://administrator:Aa123456!@192.168.100.249:5432/ingo_benchmarks".into()).await;     
    println!("Connect result: {:?}", connect_result);
    // let pool = connect_result.unwrap();
    let msm_id = zkbench::add_msm(connect_result, b).await;
    // println!("Added new ntt with id {:?}", ntt_id);
}