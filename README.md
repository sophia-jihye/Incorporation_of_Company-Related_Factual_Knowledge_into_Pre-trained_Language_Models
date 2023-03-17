# Incorporation of Company-related Factual Knowledge into Pretrained Language Models for Stock-related Spam Tweet Filtering

## Citation
```
In preparation
```

## Setup 
### `conda` environment
```bash
conda create --name transformers python=3.7
conda activate transformers
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
pip install transformers scikit-learn pandas tqdm matplotlib seaborn ipython nltk
```

### using `pip`
```bash
pip3 install torch torchvision torchaudio
pip install transformers scikit-learn pandas tqdm matplotlib seaborn ipython nltk
```


## Dataset for post-training (Form 10-Ks)

### Python packages
You can download the 3,990 item 1 sections used in the experiment <a href="https://drive.google.com/drive/folders/1wDletufalrRncQEQxRgCQlqyoMWGg8x3?usp=sharing">here</a>.
- The dataset includes item 1 sections published in 10-K filings in 2016 only. No other years are included.
- We report that we used <a href="https://sec-api.io">SEC-API.io</a> for scraping Form 10-Ks and extracting item 1 sections.


## Dataset for fine-tuning (Tweet dataset) 
* Cresci, S., Lillo, F., Regoli, D., Tardelli, S., & Tesconi, M. (2019). Cashtag Piggybacking: Uncovering Spam and Bot Activity in Stock Microblogs on Twitter. ACM Transactions on the Web (TWEB), 13(2), 11.
* [Cresci et al. (2019) dataset](https://zenodo.org/record/2686862#.Yi2D4nrP23A)
