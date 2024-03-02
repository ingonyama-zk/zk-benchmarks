use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use criterion::black_box; // Ensures computations are not optimized away
use icicle_bn254::curve::{CurveCfg, G1Affine, G1Projective, G2CurveCfg, G2Projective, ScalarCfg, ScalarField};
use icicle_core::curve::Curve;
use icicle_core::traits::GenerateRandom;
use icicle_core::msm;
use icicle_cuda_runtime::stream::CudaStream;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use zkbench::{MsmBenchmark,git_id};
use std::fs::write;
use std::env;
use chrono::{Utc, DateTime, NaiveDateTime};

// A function to benchmark
fn process_data(upper_points: &[G1Affine], upper_scalars: &[ScalarField]) -> i32 {
    // let upper_size = 100;
    let size = 10;
    let points = HostOrDeviceSlice::Host(upper_points[..size].to_vec());
    // let points = HostOrDeviceSlice::Host(upper_points.to_vec());
    let scalars = HostOrDeviceSlice::Host(upper_scalars[..size].to_vec());
    let mut msm_results: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = msm::MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
    stream.synchronize().unwrap();
    // msm_results.copy_to_host(&mut msm_host_result[..]).unwrap();
    stream.destroy().unwrap();
    1
}

// The benchmark function
fn benchmark(c: &mut Criterion) {
    let timestamp_utc: DateTime<Utc> = Utc::now();
    let test_timestamp_naive: NaiveDateTime = timestamp_utc.naive_utc();
    let repository_path = env::var("BENCHMARK_REPO").expect("BENCHMARK_REPO must be set");
    let mut b = MsmBenchmark::default();
    b.test_timestamp = test_timestamp_naive;
    b.team = "Ingonyama".to_string();
    b.project = "ICICLE".to_string();
    b.uses = "BN254".to_string();
    b.git_id = zkbench::git_id(&repository_path);
    b.frequency_mhz = 0;
    b.vector_size = 0;
    b.batch_size = 1;
    b.comment = "junk".to_string();
    let device = zkbench::gpu_name(0).trim_end_matches('\n').to_string();
    b.runs_on = device;

    let json_string = serde_json::to_string_pretty(&b).expect("Serialization failed");
    let file_path = "./metadata.json";
    write(file_path, json_string).expect("Failed to write JSON to file");
    let mut group = c.benchmark_group("ProcessData");
    group.sample_size(10); // Default is 100

    // Example sizes for the input data
    // let sizes = [10, 100, 1_000, 10_000];
    let sizes = [10, 100];

    for &size in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || {  
                    // Setup code: generate random inputs
                    let points = CurveCfg::generate_random_affine_points(size);
                    let scalars = ScalarCfg::generate_random(size);
                    (points, scalars) // Return as a tuple
                },
                |(points, scalars)| {
                    // The actual benchmarked code
                    // We wrap our function call with `black_box` to prevent compiler optimizations
                    black_box(process_data(&points, &scalars))
                },
                criterion::BatchSize::SmallInput, // Adjust based on the relative expense of the setup code
            );
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
