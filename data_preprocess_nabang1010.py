import pandas as pd
from tqdm import tqdm
import random
import yaml
import os
import csv


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# def behaviors_file_process(source, target, user2int_path):


def behaviors_file_process(source, target, user2int_path):
    print("Processing", source)
    # Read behaviors.tsv
    df_behaviors = pd.read_table(
        source,
        header=None,
        names=["impression_id", "user", "time", "clicked_news", "impressions"],
    )

    # Fill NaN with space
    df_behaviors.clicked_news.fillna(" ", inplace=True)

    # Split impressions to list
    df_behaviors.impressions = df_behaviors.impressions.str.split()

    # conver user_id to int ID
    # U80234 --> 1
    user2int = {}
    for row in df_behaviors.itertuples(index=False):
        if row.user not in user2int:
            user2int[row.user] = len(user2int) + 1
    # write user2int to file
    pd.DataFrame(user2int.items(), columns=["user", "int"]).to_csv(
        user2int_path, sep="\t", index=False
    )

    for row in df_behaviors.itertuples():
        df_behaviors.at[row.Index, "user"] = user2int[row.user]
    for row in tqdm(df_behaviors.itertuples(), desc="Balancing data"):
        positive = iter([x for x in row.impressions if x.endswith("1")])
        negative = [x for x in row.impressions if x.endswith("0")]
        random.shuffle(negative)
        negative = iter(negative)
        pairs = []
        try:
            while True:
                pair = [next(positive)]
                for _ in range(config["BASE_CONFIG"]["negative_sampling_ratio"]):
                    pair.append(next(negative))
                pairs.append(pair)
        except StopIteration:
            pass
        df_behaviors.at[row.Index, "impressions"] = pairs

    df_behaviors = (
        df_behaviors.explode("impressions")
        .dropna(subset=["impressions"])
        .reset_index(drop=True)
    )
    df_behaviors[["candidate_news", "clicked"]] = pd.DataFrame(
        df_behaviors.impressions.map(
            lambda x: (
                " ".join([e.split("-")[0] for e in x]),
                " ".join([e.split("-")[1] for e in x]),
            )
        ).tolist()
    )
    df_behaviors.to_csv(
        target,
        sep="\t",
        index=False,
        columns=["user", "clicked_news", "candidate_news", "clicked"],
    )


def news_file_process(
    source, target, category2int_path, word2int_path, entity2int_path, mode
):

    print("Processing", source)

    df_news = pd.read_table(
        source,
        header=None,
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=[
            "id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "title_entities",
            "abstract_entities",
        ],
    )
    df_news.titles_entities.fillna("[]", inplace=True)
    df_news.abstract_entities.fillna("[]", inplace=True)
    df_news.fillna(' ', inplace=True)
    
    


if __name__ == "__main__":
    DATA_RAW = "/workspace/nabang1010/LBA_NLP/Recommendation_System/DATA/dev_small/"
    DATA_DIR = "./DATA_nabang1010"
    train_dir = "train"
    val_dir = "val"
    test_dir = "test"

    source = os.path.join(DATA_RAW, "behaviors.tsv")
    target = os.path.join(DATA_DIR, train_dir, "behaviors.tsv")
    user2int_path = os.path.join(DATA_DIR, train_dir, "user2int.tsv")

    behaviors_file_process(source, target, user2int_path)
