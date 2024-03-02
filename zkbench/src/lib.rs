// src/lib.rs

use sqlx::postgres::PgPool;
use sqlx::FromRow;
use chrono::{Utc, DateTime, NaiveDateTime};
use serde::{Serialize, Deserialize};
use git2::{Repository, DescribeOptions};

//////////////////////////////////////
// MSM - Multi Scalar Multiplication
//////////////////////////////////////


// SELECT column_name, data_type, is_nullable, column_default
// FROM information_schema.columns
// WHERE table_schema = 'public' 
// AND table_name   = 'msm_benchmark';

// column_name   |          data_type          | is_nullable | column_default 
// ----------------+-----------------------------+-------------+----------------
//  id             | integer                     | NO          | 
//  team           | character varying           | YES         | 
//  project        | character varying           | YES         | 
//  test_timestamp | timestamp without time zone | YES         | 
//  git_id         | character varying           | YES         | 
//  frequency_mhz  | integer                     | YES         | 
//  vector_size    | bigint                      | YES         | 
//  coefficient_c  | integer                     | YES         | 
//  batch_size     | integer                     | YES         | 
//  runtime_sec    | real                        | YES         | 
//  power_watt     | real                        | YES         | 
//  chip_temp_c    | real                        | YES         | 
//  comment        | character varying           | YES         | 
//  runs_on        | integer                     | YES         | 
//  uses           | integer                     | YES         | 


#[derive(Debug, FromRow, Default, Serialize, Deserialize, Clone)]
pub struct MsmBenchmark {
    pub id: i32,
    pub team: String,
    pub project: String,
    pub test_timestamp: NaiveDateTime,
    pub git_id: String,
    pub frequency_mhz: i32,
    pub vector_size: i64,
    pub coefficient_c: i32,
    pub batch_size: i32,
    pub runtime_sec: Option<f32>,
    pub power_watt: f32,
    pub chip_temp_c: f32,
    pub comment: String,
    pub runs_on: String,
    pub uses: String
}

pub async fn add_msm(pool: &PgPool, benchmark: MsmBenchmark) -> anyhow::Result<i32> {
// pub async fn add_msm<T,E: std::fmt::Display>(connection_result: Result<T, E>, benchmark: MsmBenchmark) -> anyhow::Result<i32> {
    print!("===> add msm");
    let timestamp_utc: DateTime<Utc> = Utc::now();
    println!("Timestamp: {}", timestamp_utc.format("%Y-%m-%d %H:%M:%S %:z"));
    let test_timestamp_naive: NaiveDateTime = timestamp_utc.naive_utc();

    let rec = sqlx::query!(
        r#"
INSERT INTO msm_benchmark 
    (
        team, project, test_timestamp, git_id, frequency_mhz, vector_size, coefficient_c, 
        batch_size, runtime_sec, power_watt, chip_temp_c, comment, runs_on, uses
    )
VALUES 
    (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,$12,
        (SELECT id FROM hw_platform WHERE device = $13), 
        (SELECT id FROM finite_field WHERE name = $14)
    )
RETURNING id
        "#,
        benchmark.team,
        benchmark.project,
        test_timestamp_naive,
        benchmark.git_id,
        benchmark.frequency_mhz,
        benchmark.vector_size,
        benchmark.coefficient_c,
        benchmark.batch_size,
        benchmark.runtime_sec,
        benchmark.power_watt,
        benchmark.chip_temp_c,
        benchmark.comment,
        benchmark.runs_on,
        benchmark.uses
    );
    
    let r = rec.fetch_one(pool).await?;
    println!("===> add msm: {}", r.id);
    // Ok(r)
    Ok(0)
}

pub async fn list_msm(pool: &PgPool) -> anyhow::Result<()> {
    println!("===> list msm");
    let recs = sqlx::query!(
        r#"
SELECT id, team, project, test_timestamp, git_id, frequency_mhz, vector_size, coefficient_c, batch_size, runtime_sec, power_watt, chip_temp_c, comment, runs_on, uses  
FROM msm_benchmark
ORDER BY id
        "#
    )
    .fetch_all(pool)
    .await?;

    for rec in recs {
        println!(
            "- [{}] {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}",
            rec.id,
            &rec.team,
            &rec.project,
            &rec.test_timestamp,
            &rec.git_id,
            &rec.frequency_mhz,
            &rec.vector_size,
            &rec.coefficient_c,
            &rec.batch_size,
            &rec.runtime_sec,
            &rec.power_watt,
            &rec.chip_temp_c,
            &rec.comment,
            &rec.runs_on,
            &rec.uses
        );
    }

    Ok(())
}

/////////////
// Poseidon 
/////////////

#[derive(Debug, FromRow, Default)]
pub struct PoseidonBenchmark {
    pub id: i32,
    pub project: String,
    pub test_timestamp: NaiveDateTime,
    pub git_id: String,
    pub frequency_mhz: i32,
    pub tree_height: i64,
    pub batch_size: i32,
    pub runtime_sec: f32,
    pub power_watt: f32,
    pub chip_temp_c: f32,
    pub comment: String,
    pub runs_on: String,
    pub uses: String
}

pub async fn add_poseidon(pool: &PgPool, benchmark: PoseidonBenchmark) -> anyhow::Result<i32> {
    let timestamp_utc: DateTime<Utc> = Utc::now();
    // println!("Timestamp: {}", timestamp_utc.format("%Y-%m-%d %H:%M:%S %:z"));
    let test_timestamp_naive: NaiveDateTime = timestamp_utc.naive_utc();
    let rec = sqlx::query!(
        r#"
INSERT INTO poseidon_benchmark 
    (
        project, test_timestamp, git_id, frequency_mhz, tree_height,  
        batch_size, runtime_sec, power_watt, chip_temp_c, comment, runs_on, uses
    )
VALUES 
    (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
        (SELECT id FROM hw_platform WHERE device = $11), 
        (SELECT id FROM finite_field WHERE name = $12)
    )
RETURNING id
        "#,
        benchmark.project,
        test_timestamp_naive,
        benchmark.git_id,
        benchmark.frequency_mhz,
        benchmark.tree_height,
        benchmark.batch_size,
        benchmark.runtime_sec,
        benchmark.power_watt,
        benchmark.chip_temp_c,
        benchmark.comment,
        benchmark.runs_on,
        benchmark.uses
    )
    .fetch_one(pool)
    .await?;
    Ok(rec.id)
}

pub async fn list_poseidon(pool: &PgPool) -> anyhow::Result<()> {
    let recs = sqlx::query!(
        r#"
SELECT id, project, test_timestamp, git_id, frequency_mhz, tree_height, batch_size, runtime_sec, power_watt, chip_temp_c, comment, runs_on, uses  
FROM poseidon_benchmark
ORDER BY id
        "#
    )
    .fetch_all(pool)
    .await?;

    for rec in recs {
        println!(
            "- [{}] {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}",
            rec.id,
            &rec.project,
            &rec.test_timestamp,
            &rec.git_id,
            &rec.frequency_mhz,
            &rec.tree_height,
            &rec.batch_size,
            &rec.runtime_sec,
            &rec.power_watt,
            &rec.chip_temp_c,
            &rec.comment,
            &rec.runs_on,
            &rec.uses
        );
    }
    Ok(())
}


/////////////////////////////////////
// NTT - Number Theoretic Transform
/////////////////////////////////////

#[derive(Debug, FromRow, Default, Serialize, Deserialize)]
pub struct NttBenchmark {
    // pub id: i32,
    pub project: String,
    pub test_timestamp: Option<NaiveDateTime>,
    pub git_id: String,
    pub frequency_mhz: i32,
    pub vector_size: i64,
    pub batch_size: i32,
    pub runtime_sec: Option<f32>,
    pub power_watt: Option<f32>,
    pub chip_temp_c: Option<f32>,
    pub comment: String,
    pub runs_on: String,
    pub uses: String
}

pub async fn add_ntt(pool: &PgPool, benchmark: NttBenchmark) -> anyhow::Result<i32> {
    let timestamp_utc: DateTime<Utc> = Utc::now();
    let test_timestamp_naive: NaiveDateTime = timestamp_utc.naive_utc();
    let rec = sqlx::query!(
        r#"
INSERT INTO ntt_benchmark 
    (
        project, test_timestamp, git_id, frequency_mhz, vector_size,  
        batch_size, runtime_sec, power_watt, chip_temp_c, comment, runs_on, uses
    )
VALUES 
    (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
        (SELECT id FROM hw_platform WHERE device = $11), 
        (SELECT id FROM finite_field WHERE name = $12)
    )
RETURNING id
        "#,
        benchmark.project,
        test_timestamp_naive,
        benchmark.git_id,
        benchmark.frequency_mhz,
        benchmark.vector_size,
        benchmark.batch_size,
        benchmark.runtime_sec,
        benchmark.power_watt,
        benchmark.chip_temp_c,
        benchmark.comment,
        benchmark.runs_on,
        benchmark.uses
    )
    .fetch_one(pool)
    .await?;
    Ok(rec.id)
}

pub async fn list_ntt(pool: &PgPool) -> anyhow::Result<()> {
    let recs = sqlx::query!(
        r#"
SELECT 
    id, project, test_timestamp, git_id, frequency_mhz, vector_size, 
    batch_size, runtime_sec, power_watt, chip_temp_c, comment, runs_on, uses  
FROM 
    ntt_benchmark
ORDER BY id
        "#
    )
    .fetch_all(pool)
    .await?;

    for rec in recs {
        println!(
            "- [{}] {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}: {:?}",
            rec.id,
            &rec.project,
            &rec.test_timestamp,
            &rec.git_id,
            &rec.frequency_mhz,
            &rec.vector_size,
            &rec.batch_size,
            &rec.runtime_sec,
            &rec.power_watt,
            &rec.chip_temp_c,
            &rec.comment,
            &rec.runs_on,
            &rec.uses
        );
    }
    Ok(())
}

//////////////////////
// Utility functions
//////////////////////

pub fn git_id(directory: &str) -> String {
    // Open the Git repository
    let repo = Repository::open(directory).expect("Failed to open repository");

    // Parse the "HEAD" revision
    let head_revision = repo.revparse_single("HEAD").expect("Failed to parse HEAD revision");

    // Get the commit ID of the parsed revision
    let commit_id = head_revision.id();
    // println!("Current commit hash: {}", commit_id);
    // commit_id.to_string()
    commit_id.to_string()[..7].to_string()
}

pub fn gpu_name(_gpu_num: u32) -> String {
    // call nvidia-smi to get the GPU name
    let output = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .expect("Failed to execute command");
    let output_string = String::from_utf8_lossy(&output.stdout).to_string();
    output_string
}