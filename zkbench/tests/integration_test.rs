use sqlx::postgres::PgPool;
use serde_json;

// #[tokio::test]
async fn test_list_msm() {
    println!("Listing of all msm");
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let connect_result = PgPool::connect(database_url).await; 
    if let Ok(pool) = connect_result {
        // Use the 'pool' variable here
        if let Err(error) = zkbench::list_msm(&pool).await {
            eprintln!("Error while listing MSM: {:?}", error);
        }
    } else {
        if let Err(error) = connect_result {
            eprintln!("Error connecting to the database: {:?}", error);
        }
    }
}



use zkbench::MsmBenchmark;
#[tokio::test]
async fn test_add_msm()  {
    let mut benchmark = MsmBenchmark::default();
    benchmark.project = "junk".to_string();
    benchmark.git_id = "abcdef12345".to_string();
    benchmark.frequency_mhz = 250;
    benchmark.vector_size = 1024;
    benchmark.coefficient_c = 12;
    benchmark.batch_size = 64;
    // benchmark.runtime_sec =1.23;
    benchmark.power_watt = 50.3;
    benchmark.chip_temp_c = 70.1;
    benchmark.comment = "zkbench".to_string();
    benchmark.runs_on = "RTX 4090".to_string();
    benchmark.uses = "BLS12-377".to_string();
    // let json_string = serde_json::to_string(&benchmark).expect("Serialization failed");
    // println!("{}", json_string);
    // let json_string = r#"{"id":0,"project":"junk","test_timestamp":"1970-01-01T00:00:00","git_id":"abcdef12345","frequency_mhz":250,"vector_size":1024,"coefficient_c":12,"batch_size":64,"power_watt":50.3,"chip_temp_c":70.1,"comment":"zkbench","runs_on":"RTX 4090","uses":"BLS12-377"}"#;
    // let json_string = r#"{"id":0,"project":"junk","test_timestamp":"1970-01-01T00:00:00","git_id":"abcdef12345","frequency_mhz":250,"vector_size":1024,"coefficient_c":12,"batch_size":64,"runtime_sec":1.23,"power_watt":50.3,"chip_temp_c":70.1,"comment":"zkbench","runs_on":"RTX 4090","uses":"BLS12-377"}"#;
    // let b: MsmBenchmark = serde_json::from_str(json_string).expect("Deserialization failed");
    // println!("Stas {:?}", b);
    
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let connect_result = PgPool::connect(database_url).await;     
    let pool = connect_result.unwrap();
    let msm_id = zkbench::add_msm(&pool, benchmark).await;
    println!("Added new msm with id {:?}", msm_id);
}

// #[tokio::test]
async fn test_list_poseidon() {
    println!("Listing of all poseidon");
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let connect_result = PgPool::connect(database_url).await; 
    let pool = connect_result.unwrap();
    zkbench::list_poseidon(&pool).await;
}

use zkbench::PoseidonBenchmark;

// #[tokio::test]
async fn test_add_poseidon()  {
    let mut benchmark = PoseidonBenchmark::default();
    benchmark.project = "junk".to_string();
    benchmark.git_id = "abcdef12345".to_string();
    benchmark.frequency_mhz = 250;
    benchmark.tree_height = 5;
    benchmark.batch_size = 64;
    benchmark.runtime_sec =1.23;
    benchmark.power_watt = 50.3;
    benchmark.chip_temp_c = 70.1;
    benchmark.comment = "zkbench".to_string();
    benchmark.runs_on = "RTX 4090".to_string();
    benchmark.uses = "BLS12-377".to_string();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let connect_result = PgPool::connect(database_url).await;     
    let pool = connect_result.unwrap();
    let poseidon_id = zkbench::add_poseidon(&pool, benchmark).await;
    println!("Added new poseidon with id {:?}", poseidon_id);
}

// #[tokio::test]
async fn test_list_ntt() {
    println!("Listing of all ntt");
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let connect_result = PgPool::connect(database_url).await; 
    let pool = connect_result.unwrap();
    zkbench::list_ntt(&pool).await;
}


use zkbench::NttBenchmark;
// #[tokio::test]
async fn test_add_ntt()  {
    let mut benchmark = NttBenchmark::default();
    benchmark.project = "junk".to_string();
    benchmark.git_id = "abcdef12345".to_string();
    benchmark.frequency_mhz = 250;
    benchmark.vector_size = 1024;
    benchmark.batch_size = 64;
    benchmark.runtime_sec =1.23;
    benchmark.power_watt = 50.3;
    benchmark.chip_temp_c = 70.1;
    benchmark.comment = "zkbench".to_string();
    benchmark.runs_on = "RTX 4090".to_string();
    benchmark.uses = "BLS12-377".to_string();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let connect_result = PgPool::connect(database_url).await;     
    let pool = connect_result.unwrap();
    let ntt_id = zkbench::add_ntt(&pool, benchmark).await;
    println!("Added new poseidon with id {:?}", ntt_id);
}
