use std::collections::HashMap;

use colored::{Color, Colorize};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Merge {
    rank: u32,
    pair: [u32; 2],
    new_token: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MergePriority {
    level: u8,
    rank: u32,
    new_token: u32,
}

impl MergePriority {
    const DEFAULT: Self = Self {
        level: 0,
        rank: u32::MAX,
        new_token: u32::MAX,
    };

    fn is_default(&self) -> bool {
        self.rank == u32::MAX
    }
}

impl MergePriority {
    fn from_merge(merge: Merge, level: u8) -> Self {
        MergePriority {
            level,
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

pub struct FastBPETokenizer {
    tokens: Vec<Vec<u8>>,
    byte_to_token: [u32; 256],
    single_pass_merges: FxHashMap<[u32; 2], MergePriority>,
}

impl FastBPETokenizer {
    pub fn load_from_bytes(bytes: &[u8]) -> Self {
        let json = serde_json::from_slice::<Value>(bytes).unwrap();
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

        let single_pass_merges: FxHashMap<[u32; 2], MergePriority> = current_pass_merges
            .drain(..)
            .enumerate()
            .flat_map(|(priority, v)| {
                let merges = &merges;
                v.into_iter().map(move |i| {
                    let merge = merges[i];
                    (merge.pair, MergePriority::from_merge(merge, priority as u8))
                })
            })
            .collect();

        let mut byte_to_token = [u32::MAX; 256];
        for byte in 0..255 {
            if let Some(token) = tokens.iter().position(|v| v == &[byte]) {
                byte_to_token[byte as usize] = token as u32;
            }
        }

        Self {
            byte_to_token,
            single_pass_merges,
            tokens,
        }
    }
}

pub struct LayersUsed([usize; 256]);

impl LayersUsed {
    fn new() -> Self {
        Self([usize::MAX; 256])
    }

    fn set(&mut self, layer: u8, index: usize) {
        let mutable = unsafe { self.0.get_unchecked_mut(index) };
        *mutable = (*mutable).min(layer as usize);
    }

    fn first_used_index(&self, layer: u8) -> usize {
        unsafe { *self.0.get_unchecked(layer as usize) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TokenData {
    token: u32,
    merge: MergePriority,
}

impl TokenData {
    pub const DEFAULT: Self = Self {
        token: u32::MAX,
        merge: MergePriority::DEFAULT,
    };

    pub fn token(&self) -> u32 {
        self.token
    }
}

pub struct MergeQueue {
    resolved_index: usize,
    buffer_index: usize,
    buffer: [MergePriority; 10],
    first: u32,
    last_unchanged_from_level: bool,
}

impl Default for MergeQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeQueue {
    pub fn new() -> Self {
        Self {
            resolved_index: 0,
            buffer_index: 0,
            buffer: [MergePriority::DEFAULT; 10],
            first: u32::MAX,
            last_unchanged_from_level: false,
        }
    }

    fn add_unprocessed(
        &mut self,
        tokens: &mut [TokenData],
        token: TokenData,
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
    ) {
        Self::add_unprocessed_raw_and_maybe_calculate_merge(
            tokens,
            &mut self.resolved_index,
            token,
            merges_map,
            layers_used,
            self.last_unchanged_from_level,
        );
    }

    fn add_unprocessed_raw_and_maybe_calculate_merge(
        tokens: &mut [TokenData],
        resolved_index: &mut usize,
        token: TokenData,
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
        last_unchanged_from_level: bool,
    ) {
        if last_unchanged_from_level {
            Self::add_unprocessed_raw(tokens, resolved_index, token);
        } else {
            Self::add_unprocessed_raw_and_calculate_merge(
                tokens,
                resolved_index,
                token.token,
                merges_map,
                layers_used,
            );
        }
    }

    fn add_unprocessed_raw_and_calculate_merge(
        tokens: &mut [TokenData],
        resolved_index: &mut usize,
        token: u32,
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
    ) {
        let index = *resolved_index - 1;
        let priority = if let Some(last) = tokens.get(index) {
            match merges_map.get(&[last.token, token]) {
                Some(merge) => {
                    layers_used.set(merge.level, index);
                    *merge
                }
                None => MergePriority::DEFAULT,
            }
        } else {
            MergePriority::DEFAULT
        };

        Self::add_unprocessed_raw(
            tokens,
            resolved_index,
            TokenData {
                token,
                merge: priority,
            },
        );
    }

    fn add_unprocessed_raw(tokens: &mut [TokenData], resolved_index: &mut usize, token: TokenData) {
        unsafe {
            *tokens.get_unchecked_mut(*resolved_index) = token;
        }
        *resolved_index += 1;
    }

    fn push_buffer(&mut self, merge: MergePriority, first: u32) {
        if self.buffer_index == 0 {
            self.first = first;
        }
        unsafe {
            *self.buffer.get_unchecked_mut(self.buffer_index) = merge;
        }
        self.buffer_index += 1;
    }

    pub fn resolve(
        &mut self,
        tokens: &mut [TokenData],
        bytes: &[u8],
        tokenizer: &FastBPETokenizer,
    ) {
        assert!(tokens.len() >= bytes.len());
        let mut buffer_processed_end = tokens.len();
        let mut layers_used = LayersUsed::new();
        let mut max_pass = 0;
        for (i, b) in bytes.iter().enumerate() {
            let token = unsafe { *tokenizer.byte_to_token.get_unchecked(*b as usize) };
            let data = TokenData {
                token,
                merge: if let Some(&next) = bytes.get(i + 1) {
                    match tokenizer.single_pass_merges.get(&[token, unsafe {
                        *tokenizer.byte_to_token.get_unchecked(next as usize)
                    }]) {
                        Some(merge) => {
                            max_pass = max_pass.max(merge.level);
                            layers_used.set(merge.level, i);
                            *merge
                        }
                        None => MergePriority::DEFAULT,
                    }
                } else {
                    MergePriority::DEFAULT
                },
            };
            unsafe {
                *tokens.get_unchecked_mut(i) = data;
            }
        }

        for i in 0..=max_pass {
            println!("buffer_processed_end: {buffer_processed_end}");
            self.resolve_level(
                &mut buffer_processed_end,
                tokens,
                &tokenizer.single_pass_merges,
                &mut layers_used,
                i,
            );
        }
    }

    fn resolve_level(
        &mut self,
        buffer_processed_end: &mut usize,
        tokens: &mut [TokenData],
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
        level: u8,
    ) {
        self.last_unchanged_from_level = true;

        assert!(*buffer_processed_end <= tokens.len());
        if *buffer_processed_end <= 1 {
            return;
        }

        let start = layers_used.first_used_index(level);
        println!("start: {start} vs {buffer_processed_end}");
        self.resolved_index = start.saturating_sub(1);
        self.buffer_index = start;
        if start >= *buffer_processed_end {
            return;
        }

        for read_index in start..*buffer_processed_end - 1 {
            let this_token_index = read_index;
            let next_token_index = read_index + 1;
            let current_token = unsafe { *tokens.get_unchecked(this_token_index) };
            let next_token = unsafe { *tokens.get_unchecked(next_token_index) };
            let merge = next_token.merge;
            // If the level of the merge is not the current level, do not merge yet
            if merge.level != level || merge.is_default() {
                // Flush the merge buffer and add the current token unprocessed to the buffer
                if self.buffer_index == 0 {
                    self.add_unprocessed(tokens, current_token, merges_map, layers_used);
                    self.last_unchanged_from_level = true;
                } else {
                    self.flush(tokens, merges_map, layers_used);
                }
            } else {
                // If the new merge is a lower rank than the previous merge, do the previous merge instead
                match self.buffer.get(self.buffer_index - 1) {
                    Some(prev_merge) if prev_merge.rank < merge.rank => {
                        // Flush the merge buffer
                        self.flush(tokens, merges_map, layers_used);
                    }
                    _ => {
                        // Otherwise add the merge to the buffer
                        self.push_buffer(merge, current_token.token);
                    }
                }
            }
        }

        // Just add the last token to the buffer unprocessed
        if self.buffer_index == 0 {
            self.add_unprocessed(
                tokens,
                unsafe { *tokens.get_unchecked(*buffer_processed_end - 1) },
                merges_map,
                layers_used,
            );
            self.last_unchanged_from_level = true;
        } else {
            self.flush(tokens, merges_map, layers_used);
        }

        *buffer_processed_end = self.resolved_index;
    }

    fn flush(
        &mut self,
        tokens: &mut [TokenData],
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
    ) {
        let len = self.buffer_index;
        if len == 0 {
            return;
        }

        let even_len = len % 2 == 0;
        if even_len {
            Self::add_unprocessed_raw_and_calculate_merge(
                tokens,
                &mut self.resolved_index,
                self.first,
                merges_map,
                layers_used,
            );
            self.last_unchanged_from_level = true;
        }
        let mut index = even_len as usize;
        while index < self.buffer_index {
            let merge = unsafe { self.buffer.get_unchecked(index) };
            let token = merge.new_token;
            Self::add_unprocessed_raw_and_calculate_merge(
                tokens,
                &mut self.resolved_index,
                token,
                merges_map,
                layers_used,
            );
            self.last_unchanged_from_level = false;
            index += 2;
        }
        self.buffer_index = 0;
    }
}

pub fn pretty_print_tokens(resolved: impl Iterator<Item = u32>, tokenizer: &FastBPETokenizer) {
    let colors = [
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
    ];
    let mut i = 0;
    for token in resolved.map(|t| std::str::from_utf8(&tokenizer.tokens[t as usize]).unwrap()) {
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
