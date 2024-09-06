A fast BPE tokenizer written in Rust.

## Fast on small inputs

After pre-tokenization splitting, most inputs will be very small. FastBPE is absurdly fast on small inputs.

![Screenshot 2024-09-05 at 8 01 24 PM](https://github.com/user-attachments/assets/cb8ee307-dafb-4199-acdd-3495e7c3e8d0)

## Fast on giant inputs

Even if you don't  pre-tokenize, FastBPE is takes linear time for any input size. This makes it very fast on giant inputs.

![Screenshot 2024-09-05 at 8 01 44 PM](https://github.com/user-attachments/assets/948e1d36-8bdc-40b8-a19c-f7b56e0a03a9)
