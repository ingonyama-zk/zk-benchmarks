use criterion::{criterion_group, Criterion, BenchmarkId};
use criterion::black_box; // Ensures computations are not optimized away
use icicle_bn254::curve::{CurveCfg, G1Affine, G1Projective, G2CurveCfg, G2Affine, G2Projective, ScalarCfg, ScalarField};
use icicle_core::curve::Curve;
use icicle_core::traits::GenerateRandom;
use icicle_core::msm;
use icicle_cuda_runtime::stream::CudaStream;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use zkbench::{MsmBenchmark,git_id,gpu_name};
// use std::fs;
// use std::fs::OpenOptions;
use std::fs::File;
// use std::io::Write;
use std::env;
use chrono::{Utc, DateTime, NaiveDateTime};

// on-host G1
fn msm1(points: &[G1Affine], scalars: &[ScalarField]) -> i32 {
    let size  = points.len();
    let p = HostOrDeviceSlice::Host(points[..size].to_vec());
    let s = HostOrDeviceSlice::Host(scalars[..size].to_vec());
    let mut msm_results: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = msm::MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm::msm(&s, &p, &cfg, &mut msm_results).unwrap();
    stream.synchronize().unwrap();
    // msm_results.copy_to_host(&mut msm_host_result[..]).unwrap();
    stream.destroy().unwrap();
    1
}

// on-host G2
fn msm2(points: &[G2Affine], scalars: &[ScalarField]) -> i32 {
    let size  = points.len();
    let p = HostOrDeviceSlice::Host(points[..size].to_vec());
    let s = HostOrDeviceSlice::Host(scalars[..size].to_vec());
    let mut msm_results: HostOrDeviceSlice<'_, G2Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = msm::MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm::msm(&s, &p, &cfg, &mut msm_results).unwrap();
    stream.synchronize().unwrap();
    // msm_results.copy_to_host(&mut msm_host_result[..]).unwrap();
    stream.destroy().unwrap();
    1
}


// The benchmark function: on-host G1
fn benchmark1(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust:on-host-data:G1");
    group.sample_size(10);

    // let sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824];
    let sizes = [1024, 4096 ];
    for &size in &sizes {
        group.sample_size(10);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || {  
                    let points = CurveCfg::generate_random_affine_points(size);
                    let scalars = ScalarCfg::generate_random(size);
                    (points, scalars)
                },
                |(points, scalars)| {
                    black_box(msm1(&points, &scalars))
                },
                criterion::BatchSize::SmallInput, // Adjust based on the relative expense of the setup code
            );
        });
    }
    group.finish();
}

// The benchmark function: on-host G2
fn benchmark2(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust:on-host-data:G2");
    group.sample_size(10);
    
    // let sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824];
    let sizes = [1024, 4096 ];

    for &size in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || {  
                    let points = G2CurveCfg::generate_random_affine_points(size);
                    let scalars = ScalarCfg::generate_random(size);
                    (points, scalars)
                },
                |(points, scalars)| {
                    black_box(msm2(&points, &scalars))
                },
                criterion::BatchSize::SmallInput, // Adjust based on the relative expense of the setup code
            );
        });
    }
    group.finish();
}

criterion_group!(group1, benchmark1);
criterion_group!(group2, benchmark2);

fn main() {
    let metadata_path = "/tmp/metadata.json";
    let mut metadata: Vec<MsmBenchmark> = Vec::new();

    // common metadata for benchmark groups
    let timestamp_utc: DateTime<Utc> = Utc::now();
    let test_timestamp_naive: NaiveDateTime = timestamp_utc.naive_utc();
    let mut m = MsmBenchmark::default();
    m.test_timestamp = test_timestamp_naive;
    m.team = "Ingonyama".to_string();
    m.project = "ICICLE".to_string();
    m.uses = "BN254".to_string();
    let repository_path = env::var("BENCHMARK_REPO").expect("BENCHMARK_REPO must be set");
    m.git_id = git_id(&repository_path);
    m.frequency_mhz = 0; // TODO: get the frequency from the system
    m.vector_size = 0;
    m.batch_size = 1;
    // let device = gpu_name(0);
    m.runs_on = gpu_name(0);

    println!("Running group1 benchmarks...");
    // m.comment = "junk".to_string();
    metadata.push(m.clone());
    group1();
    
    println!("Running group2 benchmarks...");
    // m.comment = "junk2".to_string();
    metadata.push(m.clone());
    group2();

    println!("Writing metadata json to file {}", metadata_path);
    let file = File::create(metadata_path).expect("Failed to create file");
    serde_json::to_writer(file, &metadata).expect("Failed to write JSON to file");
}