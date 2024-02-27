use std::fs;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zkbench::{MsmBenchmark,NttBenchmark,git_id};

pub fn fibonacci(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;

    match n {
        0 => b,
        _ => {
            for _ in 0..n {
                let c = a + b;
                a = b;
                b = c;
            }
            b
        }
    }
}



pub fn criterion_benchmark(c: &mut Criterion) {
    
    let mut b = NttBenchmark::default();
    b.project = "junk".to_string();
    let repository_path = env::var("BENCHMARK_REPO").expect("BENCHMARK_REPO must be set");
    b.git_id = zkbench::git_id(repository_path);
    b.frequency_mhz = 125;
    b.vector_size = 134217728;
    b.batch_size = 3;
    b.comment = "example-benchmark".to_string();
    b.runs_on = "vu35p (C1100 board)".to_string();
    b.uses = "BLS12-381".to_string();
    let json_string = serde_json::to_string_pretty(&b).expect("Serialization failed");
    let file_path = "/tmp/metadata.json";
    std::fs::write(file_path, json_string).expect("Failed to write JSON to file");
    println!("JSON metadata saved to file: {}", file_path);
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
