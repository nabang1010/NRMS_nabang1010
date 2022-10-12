import pandas as pd
from tqdm import tqdm
import random
import yaml
import os
import csv
import json
from nltk.tokenize import word_tokenize
import swifter

# ====================================================================================

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# ====================================================================================


def row_process(row, category2int, entity2int, word2int):
    new_row = [
        row.id,
        category2int[row.category] if row.category in category2int else 0,
        category2int[row.subcategory] if row.subcategory in category2int else 0,
        [0] * config["BASE_CONFIG"]["num_words_title"],
        [0] * config["BASE_CONFIG"]["num_words_abstract"],
        [0] * config["BASE_CONFIG"]["num_words_title"],
        [0] * config["BASE_CONFIG"]["num_words_abstract"],
    ]
    local_entity_map = {}
    for e in json.loads(row.title_entities):
        if (
            e["Confidence"] > config["BASE_CONFIG"]["entity_confidence_threshold"]
            and e["WikidataId"] in entity2int
        ):
            for x in " ".join(e["SurfaceForms"]).lower().split():
                local_entity_map[x] = entity2int[e["WikidataId"]]
    for e in json.loads(row.abstract_entities):
        if (
            e["Confidence"] > config["BASE_CONFIG"]["entity_confidence_threshold"]
            and e["WikidataId"] in entity2int
        ):
            for x in " ".join(e["SurfaceForms"]).lower().split():
                local_entity_map[x] = entity2int[e["WikidataId"]]

    try:
        for i, w in enumerate(word_tokenize(row.title.lower())):
            if w in word2int:
                new_row[3][i] = word2int[w]
                if w in local_entity_map:
                    new_row[5][i] = local_entity_map[w]
    except IndexError:
        pass

    try:
        for i, w in enumerate(word_tokenize(row.abstract.lower())):
            if w in word2int:
                new_row[4][i] = word2int[w]
                if w in local_entity_map:
                    new_row[6][i] = local_entity_map[w]
    except IndexError:
        pass

    return pd.Series(
        new_row,
        index=[
            "id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "title_entities",
            "abstract_entities",
        ],
    )


# ====================================================================================


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


# ====================================================================================


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
    df_news.fillna(" ", inplace=True)

    if mode == "train":
        category2int = {}
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}
        for row in df_news.itertuples(index=False):

            # gộp cả category và subcategory vào category2int

            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in category2int:
                category2int[row.subcategory] = len(category2int) + 1

            # ----- đếm số lần xuất hiện của các từ trong title và abstract
            for w in word_tokenize(
                row.title.lower()
            ):  # tokenize title đã được chuyển thành chữ thường
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

            for w in word_tokenize(
                row.abstract.lower()
            ):  # tokenize abstract đã được chuyển thành chữ thường
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

            for e in json.loads(row.title_entities):
                times = len(e["OccurrenceOffsets"]) * e["Confidence"]
                if times > 0:
                    if e["WikidataId"] not in entity2freq:
                        entity2freq[e["WikidataId"]] = times
                    else:
                        entity2freq[e["WikidataId"]] += times

            for e in json.loads(row.abstract_entities):
                times = len(e["OccurrenceOffsets"]) * e["Confidence"]
                if times > 0:
                    if e["WikidataId"] not in entity2freq:
                        entity2freq[e["WikidataId"]] = times
                    else:
                        entity2freq[e["WikidataId"]] += times

        for k, v in word2freq.items():
            if v >= config["BASE_CONFIG"]["word_freq_threshold"]:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= config["BASE_CONFIG"]["entity_freq_threshold"]:
                entity2int[k] = len(entity2int) + 1

        parsed_news = df_news.swifter.apply(row_process, axis=1)

        pd.DataFrame(category2int.items(), columns=["category", "int"]).to_csv(
            category2int_path, sep="\t", index=False
        )

        pd.DataFrame(word2int.items(), columns=["word", "int"]).to_csv(
            word2int_path, sep="\t", index=False
        )
        pd.DataFrame(entity2int.items(), columns=["entity", "int"]).to_csv(
            entity2int_path, sep="\t", index=False
        )

    elif mode == "test":
        category2int = dict(pd.read_table(category2int_path).values.tolist())

        word2int = dict(pd.read_table(word2int_path, na_filter=False).values.tolist())

        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        parsed_news = df_news.swifter.apply(row_process, axis=1)

        parsed_news.to_csv(target, sep="\t", index=False)

    else:
        raise ValueError("mode must be train or test")


# ====================================================================================


# ====================================================================================

if __name__ == "__main__":
    DATA_RAW = "/workspace/nabang1010/LBA_NLP/Recommendation_System/DATA/dev_small/"
    DATA_DIR = "./DATA_nabang1010"

    train_dir = "train"
    val_dir = "val"
    test_dir = "test"

    source_behaviors = os.path.join(DATA_RAW, "behaviors.tsv")
    target_behaviors = os.path.join(DATA_DIR, train_dir, "behaviors_preprocessed.tsv")
    user2int_path = os.path.join(DATA_DIR, train_dir, "user2int.tsv")

    source_news = os.path.join(DATA_RAW, "news.tsv")
    target_news = os.path.join(DATA_DIR, train_dir, "news_preprocessed.tsv")
    category2int_path = os.path.join(DATA_DIR, train_dir, "category2int.tsv")
    word2int_path = os.path.join(DATA_DIR, train_dir, "word2int.tsv")
    entity2int_path = os.path.join(DATA_DIR, train_dir, "entity2int.tsv")

    behaviors_file_process(source_behaviors, target_behaviors, user2int_path)

    news_file_process(
        source_news,
        target_news,
        category2int_path,
        word2int_path,
        entity2int_path,
        mode="train",
    )
