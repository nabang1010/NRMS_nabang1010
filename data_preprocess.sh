python behaviors_preprocess.py
python news_preprocess.py --source_news ../../DATA/train_small/news.tsv \
                            --target_news train/news_preprocessed.tsv \
                            --category2int_path train/category2int.tsv \
                            --word2int_path train/word2int.tsv \
                            --entity2int_path train/entity2int.tsv \
                            --mode train



python news_preprocess.py --source_news ../../DATA/dev_small/news.tsv \
                            --target_news val/news_preprocessed.tsv \
                            --category2int_path train/category2int.tsv \
                            --word2int_path train/word2int.tsv \
                            --entity2int_path train/entity2int.tsv \
                            --mode test

