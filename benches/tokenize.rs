use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use fast_bpe::*;
use rand::Rng;
use tokenizers::Tokenizer;

pub fn criterion_benchmark(c: &mut Criterion) {
    let bytes = include_bytes!("../tokenizer.json");
    let tokenizer = FastBPETokenizer::load_from_bytes(bytes);
    let hf_tokenizer = Tokenizer::from_bytes(bytes).unwrap();

    // read the first argument as a file path to read from
    let mut group = c.benchmark_group("tokenize");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for size in [10, 100, 1000, 10000, 100000, 1000000, 10000000] {
        group.throughput(Throughput::Bytes(size as u64));
        group.sample_size(10.max(10000.min(10000000 / size)));
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
