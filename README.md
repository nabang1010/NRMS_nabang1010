# NRMS - Neural News Recommendation with Multi-Head Self-Attention
----

***@nabang1010***

## Data Preprocess

```console
sh data_preprocess.sh
```

### Behaviors tsv file preprocess

* input: `behaviors.tsv`
* output: `behaviors_preprocessed.tsv`, `user2int.tsv`



```console
python data_preprocess/behaviors_preprocess.py --source_behaviors ../../DATA/train_small/behaviors.tsv \
                                                --target_behaviors DATA_nabang1010/train/behaviors_preprocessed.tsv \
                                                --user2int_path DATA_nabang1010/train/user2int.tsv \
```



### News tsv file preprocess

* input: `news.tsv`
* output: `news_preprocessed.tsv`, `category2int.tsv`, `word2int.tsv`, `entity2int.tsv`

```console
python data_preprocess/news_preprocess.py --source_news ../../DATA/train_small/news.tsv \
                            --target_news DATA_nabang1010/train/news_preprocessed.tsv \
                            --category2int_path DATA_nabang1010/train/category2int.tsv \
                            --word2int_path DATA_nabang1010/train/word2int.tsv \
                            --entity2int_path DATA_nabang1010/train/entity2int.tsv \
                            --mode train
```

### Generate word embedding

* input: `glove.840B.300d.txt`, `word2int.tsv`
* output: `pretrained_word_embedding.npy`

```console
python data_preprocess/generate_word_embedding.py --pretrained_embedding ../../DATA/glove/glove.840B.300d.txt \
                                    --word2int_path DATA_nabang1010/train/word2int.tsv \
                                    --target_embedding DATA_nabang1010/train/pretrained_word_embedding.npy
```
### Generate entity embedding

* input: `entity_embedding.vec`, `entity2int.tsv`
* output: `pretrained_entity_embedding.npy`

```console

python data_preprocess/generate_entity_embedding.py --source_entity_embedding ../../DATA/train_small/entity_embedding.vec \
                                    --entity2int_path DATA_nabang1010/train/entity2int.tsv \
                                    --target_embedding DATA_nabang1010/train/pretrained_entity_embedding.npy
```

## Training




## To Do

