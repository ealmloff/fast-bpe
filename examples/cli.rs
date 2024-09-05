use std::io::Write;

fn main() {
    let bytes = include_bytes!("../tokenizer.json");
    let tokenizer = fast_bpe::FastBPETokenizer::load_from_bytes(bytes);

    loop {
        // read a line from stdin
        let mut line = String::new();
        print!("> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut line).unwrap();

        let text = line.trim();
        let bytes = text.as_bytes();
        let mut input_tokens = vec![fast_bpe::TokenData::DEFAULT; bytes.len()].into_boxed_slice();
        let mut merge_queue = fast_bpe::MergeQueue::new();

        merge_queue.resolve(&mut input_tokens, bytes, &tokenizer);
        fast_bpe::pretty_print_tokens(input_tokens.iter().map(|t| t.token()), &tokenizer);
        println!();
    }
}
