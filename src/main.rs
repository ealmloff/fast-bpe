#![feature(string_remove_matches)]
#![feature(array_windows)]
#![feature(allocator_api)]
#![feature(portable_simd)]

use std::{
    collections::HashMap,
    io::{Read, Write},
};

use colored::{Color, Colorize};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Merge {
    rank: u32,
    pair: [u32; 2],
    new_token: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct MergePriority {
    rank: u32,
    new_token: u32,
}

impl MergePriority {
    const DEFAULT: Self = Self {
        rank: u32::MAX,
        new_token: u32::MAX,
    };
}

impl From<Merge> for MergePriority {
    fn from(merge: Merge) -> Self {
        MergePriority {
            rank: merge.rank,
            new_token: merge.new_token,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SerializedModel {
    vocab: std::collections::HashMap<String, u32>,
    merges: Vec<String>,
}

// There should be something like a "max spread"
// which is the window that a merge can effect?

// [a, b, c]
// ab, bc

fn normalize_token(token: &str) -> String {
    token.replace('Ġ', " ")
}

fn main() {
    let json = serde_json::from_str::<Value>(include_str!("../tokenizer.json")).unwrap();
    let model = json["model"].clone();
    let deserialized = serde_json::from_value::<SerializedModel>(model).unwrap();

    let vocab: HashMap<_, _> = deserialized
        .vocab
        .into_iter()
        .map(|(k, v)| {
            let k = normalize_token(&k);
            (k.as_bytes().to_vec(), v)
        })
        .collect();
    let mut vocab_sorted: Vec<_> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();
    vocab_sorted.sort_by_key(|(_, v)| *v);
    let tokens: Vec<_> = vocab_sorted.into_iter().map(|(k, _)| k).collect();

    println!("There are {} tokens", tokens.len());

    let merges: Vec<_> = deserialized
        .merges
        .into_iter()
        .enumerate()
        .map(|(rank, merge)| {
            let (first, second) = merge.split_once(' ').unwrap();
            let first = normalize_token(first);
            let second = normalize_token(second);
            let first_bytes = first.as_bytes();
            let second_bytes = second.as_bytes();
            let merged: Vec<u8> = first_bytes
                .iter()
                .chain(second_bytes.iter())
                .copied()
                .collect();
            let new_token = *vocab.get(&merged).unwrap();
            debug_assert_eq!(merged, tokens[new_token as usize]);
            let first = *vocab.get(first_bytes).unwrap();
            let second = *vocab.get(second_bytes).unwrap();
            Merge {
                rank: rank as u32,
                pair: [first, second],
                new_token,
            }
        })
        .collect();
    let token_to_merge: FxHashMap<_, _> = merges
        .iter()
        .map(|merge| (merge.new_token, *merge))
        .collect();

    let mut single_pass_merges: Vec<FxHashMap<[u32; 2], MergePriority>> = Vec::new();
    let mut current_pass_merges: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    let mut tokens_used_to_create_merge = FxHashSet::default();

    'o: for (merge_idx, merge) in merges.iter().enumerate() {
        tokens_used_to_create_merge.clear();
        stack.clear();
        stack.extend(merge.pair);
        while let Some(token) = stack.pop() {
            if let Some(merge) = token_to_merge.get(&token) {
                let [first, second] = merge.pair;
                stack.push(first);
                stack.push(second);
                tokens_used_to_create_merge.insert(token);
            }
        }

        let mut index = current_pass_merges.len();
        while index > 0 {
            index -= 1;

            let mut stop_here = false;

            // Check if the new token is a prefix of any existing merge or a postfix of any existing merge. If it is, use the last match after that.
            for other in current_pass_merges[index].iter().copied() {
                let other_merge = &merges[other];
                stop_here |= other_merge.pair.contains(&merge.pair[0])
                    || other_merge.pair.contains(&merge.pair[1]);
                if tokens_used_to_create_merge.contains(&other_merge.new_token) {
                    if index < current_pass_merges.len() - 1 {
                        // If it does conflict, but we fit in at least one previous merge, add the merge to the previous merge
                        current_pass_merges[index + 1].push(merge_idx);
                    } else {
                        // Otherwise, add the merge to the current pass
                        current_pass_merges.push(vec![merge_idx]);
                    }
                    continue 'o;
                }
            }

            // If the new token would eat a token that is used by this layer, stop here. This is a conflict.
            if stop_here {
                current_pass_merges[index].push(merge_idx);
                continue 'o;
            }
        }
        if current_pass_merges.is_empty() {
            // If there are no previous merges, add the merge to the current pass
            current_pass_merges.push(vec![merge_idx]);
        } else {
            // Otherwise, add the merge to the first item
            current_pass_merges[0].push(merge_idx);
        }
    }
    single_pass_merges.extend(current_pass_merges.drain(..).map(|i| {
        i.into_iter()
            .map(|i| {
                let merge = &merges[i];
                (merge.pair, (*merge).into())
            })
            .collect()
    }));

    println!("total passes: {:?}", single_pass_merges.len());

    struct MergeQueue<'a> {
        read_index: usize,
        resolved_index: usize,
        buffer_processed_end: usize,
        tokens: &'a mut [u32],
        buffer_index: usize,
        buffer: [MergePriority; 10],
        first: u32,
    }

    impl<'a> MergeQueue<'a> {
        fn new(tokens: &'a mut [u32]) -> Self {
            Self {
                buffer_processed_end: tokens.len(),
                tokens,
                read_index: 0,
                resolved_index: 0,
                buffer_index: 0,
                buffer: [MergePriority::DEFAULT; 10],
                first: u32::MAX,
            }
        }

        fn add_unprocessed(&mut self, token: u32) {
            self.tokens[self.resolved_index] = token;
            self.resolved_index += 1;
        }

        fn push_buffer(&mut self, merge: MergePriority, first: u32) {
            if self.buffer_index == 0 {
                self.first = first;
            }
            self.buffer[self.buffer_index] = merge;
            self.buffer_index += 1;
        }

        fn resolve_level(&mut self, merges_map: &FxHashMap<[u32; 2], MergePriority>) {
            self.read_index = 0;
            self.resolved_index = 0;
            debug_assert!(self.buffer_index == 0);

            while self.read_index < self.buffer_processed_end {
                let this_token_index = self.read_index;
                let next_token_index = self.read_index + 1;
                self.read_index += 1;
                let current_token = self.tokens[this_token_index];
                let next_token = if self.read_index < self.buffer_processed_end {
                    self.tokens[next_token_index]
                } else {
                    // Just add the last token to the buffer unprocessed
                    if self.buffer_index == 0 {
                        self.add_unprocessed(current_token);
                    } else {
                        self.flush();
                    }
                    continue;
                };
                let merge = merges_map.get(&[current_token, next_token]);
                match merge {
                    Some(merge) => {
                        // If the new merge is a lower rank than the previous merge, do the previous merge instead
                        match self.buffer[..self.buffer_index].last() {
                            Some(prev_merge) if prev_merge.rank < merge.rank => {
                                // Flush the merge buffer
                                self.flush();
                            }
                            _ => {
                                // Otherwise add the merge to the buffer
                                self.push_buffer(*merge, current_token);
                            }
                        }
                    }
                    None => {
                        // Flush the merge buffer and add the current token unprocessed to the buffer
                        if self.buffer_index == 0 {
                            self.add_unprocessed(current_token);
                        } else {
                            self.flush();
                        }
                    }
                }
            }

            self.buffer_processed_end = self.resolved_index;
        }

        fn flush(&mut self) {
            let len = self.buffer_index;
            if len == 0 {
                return;
            }

            let even_len = len % 2 == 0;
            if even_len {
                self.add_unprocessed(self.first);
            }
            for merge in self.buffer[..self.buffer_index]
                .iter()
                .skip(even_len as usize)
                .step_by(2)
            {
                self.tokens[self.resolved_index] = merge.new_token;
                self.resolved_index += 1;
            }
            self.buffer_index = 0;
        }

        fn tokens(&self) -> &[u32] {
            &self.tokens[..self.resolved_index]
        }
    }

    let mut byte_to_token = vec![u32::MAX; 256];
    for byte in 0..255 {
        if let Some(token) = tokens.iter().position(|v| v == &[byte]) {
            byte_to_token[byte as usize] = token as u32;
        }
    }

    // read the first argument as a file path to read from
    let mut input = std::fs::File::open(std::env::args().nth(1).unwrap()).unwrap();
    let mut text = String::new();
    input.read_to_string(&mut text).unwrap();

    let start = std::time::Instant::now();
    let mut input_tokens: Vec<_> = text.bytes().map(|b| byte_to_token[b as usize]).collect();
    let mut merge_queue = MergeQueue::new(&mut input_tokens);
    for merges_map in &single_pass_merges {
        merge_queue.resolve_level(merges_map);
    }
    println!("time to resolve fast-bpe: {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let start = std::time::Instant::now();
    tokenizer.encode(text, true).unwrap();
    println!("time to resolve hf: {:?}", start.elapsed());
    // pretty_print_tokens(merge_queue.tokens(), &tokens);
}

fn pretty_print_tokens(resolved: &[u32], tokens: &[Vec<u8>]) {
    let colors = [
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
    ];
    let mut i = 0;
    for token in resolved
        .iter()
        .map(|t| std::str::from_utf8(&tokens[*t as usize]).unwrap())
    {
        i = (i + 1) % colors.len();
        print!("{}", token.color(colors[i]));
    }
}

// a,b e,f b,c d,ef

// | a   | b   | c   | d   | e   | f   |
// | --- | --- | --- | --- | --- | --- |
// |     | ab  | c   | d   | e   | f   |
// |     | ab  | c   | d   | ef  |     |
// |     | ab  | c   | def |     |     |
// |     |     |     |     |     |     |

// | a   | b   | c   | d   | e   | f   |
// | --- | --- | --- | --- | --- | --- |
// | a   | b   | c   | d   | ef  |     |
// | a   | b   | c   | def |     |     |
// |     | ab  | c   | def |     |     |
// |     |     |     |     |     |     |
// - Walk through the level, looking at adjacent merges
// 	- If the merge score decreases in the next merge, keep going
// 	- Otherwise, we can merge immediately?

// | a   | b   | c   | d   | e   | f   | a   |
// | --- | --- | --- | --- | --- | --- | --- |
// |     | ab  | c   | d   | ef  |     |     |
// |     | ab  | c   |     |     |     |     |
// |     |     |     |     |     |     |     |
// |     |     |     |     |     |     |     |
// Different scan levels of merges?
// - Each level applies a few merges in waves of the bpe tree
// - Small levels have a much smaller window
// 	- What is the window for the first level merge with just byte level?
// 	- 2? No because the second token might need to merge with the token after that because
// - Should work for a well defined bpe (components of merge have a higher priority than the merge itself?)

// What do "overlapping" merge rules look like?
// One merge could eat bytes that another merge may use.
// e.g. "a,b" and "b,c" are overlapping
// but "a,b" and "c,d" are not
// Prefix/postfix encoding?
