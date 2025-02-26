# PLeaS - Merging Models with Permutations and Least Squares

This repository contains the implementation for the paper "PLeaS - Merging Models with Permutations and Least Squares" and extends previous work on permutation-based model merging.

## Overview

Neural networks are not uniquely identified due to weight-space symmetries. Two networks with the same architecture but different weights might represent the same function if their weights differ only by permutations. Git Re-Basin and related approaches align these networks by finding permutations that make their weights as similar as possible.

This repository provides:

1. A modular implementation of Git Re-Basin for finding permutations between neural networks
2. An extension to partial model merging using permutation matching
3. Our novel PLeaS algorithm, which uses permutations and least squares to merge models effectively
4. Scripts to reproduce our experimental results on various datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/SewoongLab/PLeaS-Merging.git
cd PLeaS-Merging

# Install dependencies
pip install -r requirements.txt
```

See the full documentation for more details.