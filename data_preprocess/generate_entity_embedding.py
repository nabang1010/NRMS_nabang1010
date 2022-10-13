import argparse
import yaml
import numpy as np
import pandas as pd

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def generate_entity_embedding(
    source_entity_embedding, entity2int_path, target_embedding
):
    print("=======================================================================")
    print("Embedding file: {}".format(entity2int_path))
    print("----------------- Processing generate entity embedding -----------------")

    source_embedding = pd.read_table(source_entity_embedding, header=None)

    source_embedding['vector'] = source_embedding.iloc[:,
                                                       1:101].values.tolist()
    source_embedding = source_embedding[[0, 'vector'
                                         ]].rename(columns={0: "entity"})

    entity2int = pd.read_table(entity2int_path)
    merged_df = pd.merge(source_embedding, entity2int,
                         on='entity').sort_values('int')
    entity_embedding_transformed = np.random.normal(
        size=(len(entity2int) + 1, config["BASE_CONFIG"]["entity_embedding_dim"]))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector

    print(">>>>>> Saving pretrained_entity_embedding.npy <<<<<<")

    np.save(target_embedding, entity_embedding_transformed)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_entity_embedding",
        type=str,
        default="data/entity_embedding.vec",
        help="read source entity embedding entity_embedding.vec file path",
    )
    parser.add_argument(
        "--entity2int_path",
        type=str,
        default="DATA_nabang1010/train/entity2int.tsv",
        help="read entity2int.tsv file path",
    )

    parser.add_argument(
        "--target_embedding",
        type=str,
        default="DATA_nabang1010/train/pretrained_entity_embedding.npy",
        help="write target entity embedding pretrained_entity_embedding.npy file path",
    )

    args = parser.parse_args()
    generate_entity_embedding(
        args.source_entity_embedding, args.entity2int_path, args.target_embedding
    )
