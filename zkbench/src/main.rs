use std::fs;
use std::fs::File;
use serde::{Serialize, Deserialize};
use serde_json::from_str;
use serde_json::{Result,Deserializer};
use zkbench::{MsmBenchmark,NttBenchmark,git_id};
use sqlx::postgres::PgPool;
use std::env;
use dotenv::dotenv;
use std::process::{Command, Stdio};
use std::io::{self, BufRead};
use std::path::Path;


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

#[derive(Debug, Deserialize)]
pub struct Group {
    #[serde(rename = "group_name")]
    pub group_name: String,
    #[serde(rename = "benchmarks")]
    pub benchmarks: Vec<String>
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DataTypes {
    P(Performance),
    G(Group),
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // export RUST_LOG=sqlx=trace
    env_logger::init();
    // dotenv().ok();

    let repository_path = env::var("BENCHMARK_REPO").expect("BENCHMARK_REPO must be set");
    println!("Repository path: {}", repository_path);
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    println!("Database URL: {}", database_url);
    
    let id = zkbench::git_id(&repository_path);
    println!("Current commit hash: {}", id);
    // std::process::exit(1);

    let path= "../benchmarks/rust/msm";
    let output_json = File::create("/tmp/criterion.json")?;
    // let output_json = File::create("/tmp/criterion1.json").expect("Failed to create file");
    let mut child  = Command::new("cargo")
        .arg("criterion")
        .arg("--message-format")
        .arg("json")
        .current_dir(path)
        .stdout(Stdio::from(output_json))
        .spawn()?;
    // Wait for the command to complete
    let _result = child.wait()?;

    let metadata_path = "/tmp/metadata.json";
    // read metadata json into a vector of MsmBenchmark
    let metadata_file = File::open(metadata_path)?;
    let metadata: Vec<MsmBenchmark> = serde_json::from_reader(metadata_file)?;
    println!("{:#?}", metadata[0]);
    // let metadata_json = fs::read_to_string(metadata_path).expect("Failed to read the file");
    // // let mut meta: MsmBenchmark = serde_json::from_str(&metadata_json.to_string()).expect("Deserialization failed");
    // // println!("Metadata {:?}",meta);
    let connect_result = PgPool::connect(&database_url).await; 
    let pool = connect_result.unwrap();
    let data_path = Path::new("/tmp/criterion.json");
    let data_file = File::open(&data_path)?; 
    let reader = io::BufReader::new(data_file);
    let mut nof_group : usize = 0;
    let mut nof_benchmark: i32 = 0;
    for line in reader.lines() {
        let line = line?;
        let obj: Result<DataTypes> = from_str(&line);
        match obj {
            Ok(DataTypes::P(performance)) => {
                println!("Group {:?} Benchmark {:?}", nof_group, nof_benchmark);
                let id_parts: Vec<&str> = performance.benchmark_id.split('/').collect();
                let benchmark_group: &str = id_parts[0];
                let vector_size: i64 = id_parts[1].parse().unwrap();
                let runtime = normalize_runtime(performance.typical);
                // println!("Comment {:#?} vector {:#?} runtime_sec {:#?}", benchmark_group, vector_size, runtime);
                let mut b = metadata[nof_group].clone();
                b.vector_size = vector_size;
                b.runtime_sec = Some(runtime);
                b.comment = "junk ".to_string() + benchmark_group;
                println!("{:#?}", b);
                // let msm_id = zkbench::add_msm(&pool, b).await;
                // println!("Added new msm with id {:?}", msm_id);
                nof_benchmark += 1;
            }
            Ok(DataTypes::G(group)) => {
                nof_group += 1;
                nof_benchmark = 0;
                // println!("Moving to next group: {:?}", nof_group);
            }
            Err(e) => {
                eprintln!("Error deserializing Performance: {}", e);
            }
        }
    }
    pool.close().await;
    Ok(())
}