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
    dotenv().ok();

    let repository_path = env::var("BENCHMARK_REPO").expect("BENCHMARK_REPO must be set");
    println!("Repository path: {}", repository_path);
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    println!("Database URL: {}", database_url);

    // let id = zkbench::git_id(&repository_path);
    // println!("Current commit hash: {}", id);

    let path= "../benchmarks/rust/msm";
    // let output_json = File::create("/tmp/criterion.json")?;
    // let output_json = File::create("/tmp/criterion1.json").expect("Failed to create file");
    // let mut child  = Command::new("cargo")
    //     .arg("criterion")
    //     .arg("--message-format")
    //     .arg("json")
    //     .current_dir(path)
    //     .stdout(Stdio::from(output_json))
    //     .spawn()?;
    // // Wait for the command to complete
    // let _result = child.wait()?;

    // let metadata_path = "/tmp/metadata.json";
    // let metadata_json = fs::read_to_string(metadata_path).expect("Failed to read the file");
    // let mut meta: MsmBenchmark = serde_json::from_str(&metadata_json.to_string()).expect("Deserialization failed");
    // println!("Metadata {:?}",meta);

    let data_path = Path::new("../benchmarks/rust/msm/criterion.json");
    let data_file = File::open(&data_path)?; 
    let reader = io::BufReader::new(data_file);

    for line in reader.lines() {
        let line = line?;
        let obj: Result<DataTypes> = from_str(&line);
        match obj {
            Ok(DataTypes::P(performance)) => {
                println!("Performance: {:?}", performance);
            }
            Ok(DataTypes::G(group)) => {
                println!("Group: {:?}", group);
            }
            Err(e) => {
                eprintln!("Error deserializing Performance: {}", e);
            }
        }
    }

    std::process::exit(0);    
    // let performance_json = fs::read_to_string(performance_path).expect("Failed to read the file");
    // // Deserialize the first JSON object in the file
    // let mut deserializer = Deserializer::from_str(&performance_json);
    // let results: Vec<Result<Performance>> = deserializer.into_iter().collect();
    // let connect_result = PgPool::connect(&database_url).await;     
    // println!("Connect result: {:?}", connect_result);
    // let pool = connect_result.unwrap();
    // for result in results {
    //     match result {
    //         Ok(performance) => {
    //             let id_parts: Vec<&str> = performance.benchmark_id.split('/').collect();
    //             let benchmark_group: &str = id_parts[0];
    //             println!("Benchmark Group: {:?}", benchmark_group);
    //             let vector_size: i64 = id_parts[1].parse().unwrap();
    //             let mut b = meta.clone();
    //             b.vector_size = vector_size;
    //             let runtime = normalize_runtime(performance.typical);
    //             b.runtime_sec = Some(runtime);
    //             println!("Runtime: {:?}", runtime);
    //             println!("{:#?}", b);
    //             let msm_id = 0;
    //             // let msm_id = zkbench::add_msm(&pool, b).await;
    //             println!("Added new msm with id {:?}", msm_id);
    //         }
    //         Err(e) => {
    //             eprintln!("Error deserializing Performance: {}", e);
    //         }
    //     }
    // }
    // pool.close().await;
    Ok(())
}