import pandas as pd
from tqdm import tqdm
import random
import yaml

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def behaviors_file_process(source_behaviors, target_behaviors, user2int_path):
    print("Processing", source_behaviors)
    # Read behaviors.tsv
    df_behaviors = pd.read_table(
        source_behaviors,
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
        target_behaviors,
        sep="\t",
        index=False,
        columns=["user", "clicked_news", "candidate_news", "clicked"],
    )


