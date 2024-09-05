A fast BPE tokenizer written in Rust.

## Fast on small inputs

After pre-tokenization splitting, most inputs will be very small. FastBPE is absurdly fast on small inputs.

![Screenshot 2024-09-05 at 1 52 49 PM](https://github.com/user-attachments/assets/95b01835-a81c-4583-945b-caf3d12fb286)


## Fast on giant inputs

Even if you don't  pre-tokenize, FastBPE is takes linear time for any input size. This makes it very fast on giant inputs.

![Screenshot 2024-09-05 at 1 52 56 PM](https://github.com/user-attachments/assets/3caa9e2f-4e08-4ad6-8d4a-3535d6503e9f)
