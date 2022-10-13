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

    source_embedding = pd.read_table()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_entity_embedding",
        type=str,
        default="data/entity_embedding.vec",
        help="source entity embedding file",
    )
    parser.add_argument(
        "--entity2int_path",
        type=str,
        default="DATA_nabang1010/train/pretrained_entity_embedding.txt",
        help="entity2int file",
    )

    parser.add_argument(
        "--target_embedding",
        type=str,
        default="DATA_nabang1010/train/entity_embedding.npy",
    )

    args = parser.parse_args()
    generate_entity_embedding(
        args.source_entity_embedding, args.entity2int_path, args.target_embedding
    )
