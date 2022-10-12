# NRMS - Neural News Recommendation with Multi-Head Self-Attention
----

***@nabang1010***

## Data Preprocess

### Behaviors tsv file preprocess

* input: `behaviors.tsv`
* output: `behaviors_preprocessed.tsv`, `user2int.tsv`



```console
python behaviros_preprocess.py
```



### News tsv file preprocess

* input: `news.tsv`
* output: `news_preprocessed.tsv`, `category2int.tsv`, `word2int.tsv`, `entity2int.tsv`

```console
python news_preprocess.py
```

### Generate word embedding

* input: `glove.840B.300d.txt`, `word2int.tsv`
* output: `pretrained_word_embedding.npy`

```console
python generate_word_embedding.py
```

### Generate entity embedding
## To Do

