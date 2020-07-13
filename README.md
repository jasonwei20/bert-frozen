# Simple Text Classification with BERT-avgpool

This is actually the code for two projects, one on data augmentation, which I wrote a paper for, and one on gradient mining, which didn't work out because it was actually a long winded way of using a MLP. 

`code/{1,2,3}` is all for data augmentation. Most of the other code is for gradient mining.

You can get the batch-level gradient directly from PyTorch.

For getting per-example gradients, the only library that I know of that works is [autograd-hacks](https://github.com/cybertronai/autograd-hacks). 