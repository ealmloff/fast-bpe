A fast BPE tokenizer written in Rust.

## Fast on small inputs

After pre-tokenization splitting, most inputs will be very small. FastBPE is absurdly fast on small inputs.

## Fast on giant inputs

Even if you don't  pre-tokenize, FastBPE is takes linear time for any input size. This makes it very fast on giant inputs.

