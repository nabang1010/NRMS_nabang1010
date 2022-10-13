python behaviors_preprocess.py --source_behaviors ../../DATA/train_small/behaviors.tsv \
                                --target_behaviors DATA_nabang1010/train/behaviors_preprocessed.tsv \
                                --user2int_path DATA_nabang1010/train/user2int.tsv \



python news_preprocess.py --source_news ../../DATA/train_small/news.tsv \
                            --target_news DATA_nabang1010/train/news_preprocessed.tsv \
                            --category2int_path DATA_nabang1010/train/category2int.tsv \
                            --word2int_path DATA_nabang1010/train/word2int.tsv \
                            --entity2int_path DATA_nabang1010/train/entity2int.tsv \
                            --mode train



python news_preprocess.py --source_news ../../DATA/dev_small/news.tsv \
                            --target_news DATA_nabang1010/val/news_preprocessed.tsv \
                            --category2int_path DATA_nabang1010/train/category2int.tsv \
                            --word2int_path DATA_nabang1010/train/word2int.tsv \
                            --entity2int_path DATA_nabang1010/train/entity2int.tsv \
                            --mode test




