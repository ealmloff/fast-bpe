use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use fast_bpe::*;
use rand::Rng;
use tokenizers::Tokenizer;

pub fn criterion_benchmark(c: &mut Criterion) {
    const FAST_FILE: &str = "tokenizer-fast.bin";
    const HF_FILE: &str = "tokenizer.json";
    let bytes = std::fs::read(HF_FILE).unwrap();
    let tokenizer = if let Ok(tokenizer) = std::fs::read(FAST_FILE) {
        postcard::from_bytes::<fast_bpe::FastBPETokenizer>(&tokenizer).unwrap()
    } else {
        let tokenizer = fast_bpe::FastBPETokenizer::load_from_bytes(&bytes);
        std::fs::write(FAST_FILE, postcard::to_stdvec(&tokenizer).unwrap()).unwrap();
        tokenizer
    };
    let hf_tokenizer = Tokenizer::from_bytes(bytes).unwrap();

    // read the first argument as a file path to read from
    let mut group = c.benchmark_group("tokenize");
    // group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));
    let samples = 200;
    let step = samples/100;
    for size in 1..=100 {  
        let size = size * step;
        group.throughput(Throughput::Bytes(size as u64));
        // group.sample_size(10.max(10000.min(10000000 / size)));
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_millis(500));
        group.bench_with_input(BenchmarkId::new("Fast", size), &size, |b, &size| {
            let text = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(size)
                .map(char::from)
                .collect::<String>();

            let bytes = text.as_bytes();
            let mut input_tokens = vec![TokenData::DEFAULT; bytes.len()].into_boxed_slice();
            let mut merge_queue = MergeQueue::new();

            b.iter(|| merge_queue.resolve(&mut input_tokens, bytes, &tokenizer))
        });
        group.bench_with_input(BenchmarkId::new("HuggingFace", size), &size, |b, &size| {
            let text = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(size)
                .map(char::from)
                .collect::<String>();

            b.iter(|| hf_tokenizer.encode(text.clone(), true).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);