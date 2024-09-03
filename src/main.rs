#![feature(array_windows)]
#![feature(allocator_api)]
#![feature(portable_simd)]

use std::collections::HashMap;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
            assert_eq!(merged, tokens[new_token as usize]);
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

    let mut merge_depth = HashMap::new();
    fn compute_merge_depth(
        merge_depth: &mut HashMap<u32, usize>,
        token_to_merge: &FxHashMap<u32, Merge>,
        token: u32,
    ) -> usize {
        if let Some(depth) = merge_depth.get(&token) {
            return *depth;
        }

        let depth = if let Some(merge) = token_to_merge.get(&token) {
            let [first, second] = merge.pair;
            let first_range = compute_merge_depth(merge_depth, token_to_merge, first);
            let second_range = compute_merge_depth(merge_depth, token_to_merge, second);

            first_range.max(second_range) + 1
        } else {
            0
        };

        merge_depth.insert(token, depth);
        depth
    }

    let mut token_depths = vec![0; tokens.len()];
    for (i, token_depths) in token_depths.iter_mut().enumerate() {
        let depth = compute_merge_depth(&mut merge_depth, &token_to_merge, i as u32);
        *token_depths = depth;
    }

    let max = token_depths.iter().max().unwrap();
    println!("max depth: {:?}", max);
    let mut sizes = Vec::new();
    for depth in &token_depths {
        while *depth >= sizes.len() {
            sizes.push(0);
        }
        sizes[*depth] += 1;
    }
    println!("sizes");
    for (i, size) in sizes.iter().enumerate() {
        println!("{}: {}", i, size);
    }
    let sum = token_depths.iter().sum::<usize>();
    println!("avg depth: {:?}", sum as f32 / tokens.len() as f32);

    let mut merges_by_depth = vec![Vec::default(); sizes.len()];
    for (i, merge) in merges.iter().enumerate() {
        let tid = merge.new_token;
        merges_by_depth[token_depths[tid as usize]].push(i);
    }

    let mut single_pass_merges: Vec<FxHashMap<[u32; 2], MergePriority>> = Vec::new();
    let mut current_pass_merges: Vec<Vec<usize>> = Vec::new();

    'o: for (merge_idx, merge) in merges.iter().enumerate() {
        println!(
            "{:.2}% steps {}",
            (merge_idx as f32 / merges.len() as f32) * 100.,
            current_pass_merges.len()
        );

        let mut tokens_used_to_create_merge = FxHashSet::default();
        let mut stack = Vec::new();
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
        let mut ends_seen = FxHashSet::default();
        let mut starts_seen = FxHashSet::default();
        while index > 0 {
            index -= 1;

            // Check if the new token is a prefix of any existing merge or a postfix of any existing merge. If it is, use the last match after that.
            for other in current_pass_merges[index].iter().copied() {
                let other_merge = &merges[other];
                starts_seen.insert(other_merge.pair[0]);
                ends_seen.insert(other_merge.pair[1]);
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
            if starts_seen.contains(&merge.pair[1]) || ends_seen.contains(&merge.pair[0]) {
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
    println!("current_pass_merges: {:?}", current_pass_merges);
    single_pass_merges.extend(current_pass_merges.drain(..).map(|i| {
        i.into_iter()
            .map(|i| {
                let merge = &merges[i];
                (merge.pair, (*merge).into())
            })
            .collect()
    }));

    println!(
        "passes: {:?}",
        single_pass_merges
            .iter()
            .map(|x| x.len())
            .collect::<Vec<_>>()
    );
    println!("total passes: {:?}", single_pass_merges.len());

    struct MergeQueue<'a> {
        read_index: usize,
        resolved_index: usize,
        buffer_processed_end: usize,
        tokens: &'a mut [u32],
        buffer: Vec<MergePriority>,
        first: u32,
    }

    impl<'a> MergeQueue<'a> {
        fn new(tokens: &'a mut [u32]) -> Self {
            Self {
                buffer_processed_end: tokens.len(),
                tokens,
                read_index: 0,
                resolved_index: 0,
                buffer: Vec::new(),
                first: u32::MAX,
            }
        }

        fn add_unprocessed(&mut self, token: u32) {
            println!(
                "adding unprocessed token: {:?} @ {:?}",
                token, self.resolved_index
            );
            self.tokens[self.resolved_index] = token;
            self.resolved_index += 1;
        }

        fn push_buffer(&mut self, merge: MergePriority, first: u32) {
            if self.buffer.is_empty() {
                self.first = first;
            }
            self.buffer.push(merge);
        }

        fn resolve_level(&mut self, merges_map: &FxHashMap<[u32; 2], MergePriority>) {
            self.read_index = 0;
            self.resolved_index = 0;
            assert!(self.buffer.is_empty());

            while self.read_index < self.buffer_processed_end {
                let this_token_index = self.read_index;
                let next_token_index = self.read_index + 1;
                self.read_index += 1;
                let current_token = self.tokens[this_token_index];
                let next_token = if self.read_index < self.buffer_processed_end {
                    self.tokens[next_token_index]
                } else {
                    // Just add the last token to the buffer unprocessed
                    if self.buffer.is_empty() {
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
                        match self.buffer.last() {
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
                        if self.buffer.is_empty() {
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
            let len = self.buffer.len();
            if len == 0 {
                return;
            }

            let even_len = len % 2 == 0;
            if even_len {
                self.add_unprocessed(self.first);
            }
            for merge in self.buffer.iter().skip(even_len as usize).step_by(2) {
                println!(
                    "adding merge: {:?} @ {:?}",
                    merge.new_token, self.resolved_index
                );
                self.tokens[self.resolved_index] = merge.new_token;
                self.resolved_index += 1;
            }
            self.buffer.clear();
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

    loop {
        let mut text = String::new();
        std::io::stdin().read_line(&mut text).unwrap();
        let text = text.trim();
        let mut input_tokens: Vec<_> = text.bytes().map(|b| byte_to_token[b as usize]).collect();
        let mut merge_queue = MergeQueue::new(&mut input_tokens);
        for merges_map in &single_pass_merges {
            merge_queue.resolve_level(merges_map);
            println!("resolved_index: {:?}", merge_queue.tokens());
            println!(
                "detokenized: {:?}",
                merge_queue
                    .tokens()
                    .iter()
                    .map(|t| {
                        tokens
                            .get(*t as usize)
                            .map(|t| std::str::from_utf8(t).unwrap())
                    })
                    .collect::<Vec<_>>()
            );
        }
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
