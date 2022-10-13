import pandas as pd
import numpy as np
import yaml
import argparse
import csv


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def generate_word_embedding(pretrained_embedding, word2int_path, target_embedding):
    print("=======================================================================")
    print("Embedding file: {}".format(word2int_path))
    print("----------------- Processing generate word embedding -----------------")
    source_embedding = pd.read_table(
        pretrained_embedding,
        index_col=0,
        sep=" ",
        header=None,
        quoting=csv.QUOTE_NONE,
        names=range(config["BASE_CONFIG"]["word_embedding_dim"]),
    )
    word2int = pd.read_table(word2int_path, na_filter=False, index_col="word")
    source_embedding.index.rename("word", inplace=True)
    source_embedding.index.rename("word", inplace=True)

    merged = word2int.merge(
        source_embedding, how="inner", left_index=True, right_index=True
    )
    merged.set_index("int", inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1), merged.index.values)
    missed_embedding = pd.DataFrame(
        data=np.random.normal(size=(len(missed_index), config.word_embedding_dim))
    )
    missed_embedding["int"] = missed_index
    missed_embedding.set_index("int", inplace=True)

    final_embedding = pd.concat([merged, missed_embedding]).sort_index()

    print(">>>>>> Saving word_embedding.npy <<<<<<")
    np.save(target_embedding, final_embedding.values)

    print(
        f"Rate of word missed in pretrained embedding: {(len(missed_index)-1)/len(word2int):.4f}"
    )


if __name__ == "main":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_embedding",
        type=str,
        default="/workspace/nabang1010/LBA_NLP/Recommendation_System/DATA/glove/glove.840B.300d.txt",
        help="Path to source embedding pretrained glove.840B.300d.txt file  ",
    )

    parser.add_argument(
        "--word2int_path",
        type=str,
        default="data/word2int.tsv",
        help="Path to word2int.tsv file",
    )

    parser.add_argument(
        "--target_embedding",
        type=str,
        default="data/pretrained_word_embedding.npy",
        help="write pretrained_word_embedding.npy file path",
    )
    args = parser.parse_args()

    generate_word_embedding(
        args.pretrained_embedding, args.word2int_path, args.target_embedding
    )
