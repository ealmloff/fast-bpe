A fast BPE tokenizer written in Rust.

## Fast on small inputs

After pre-tokenization splitting, most inputs will be very small. FastBPE is absurdly fast on small inputs.

![Screenshot 2024-09-05 at 8 01 24 PM](https://github.com/user-attachments/assets/cb8ee307-dafb-4199-acdd-3495e7c3e8d0)

## Fast on giant inputs

Even if you don't pre-tokenize, FastBPE is takes linear time for any input size. This makes it very fast on giant inputs.

![Screenshot 2024-09-05 at 8 25 21 PM](https://github.com/user-attachments/assets/e85df113-0c7a-4547-959f-df11c7ffd891)

