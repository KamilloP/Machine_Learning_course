import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Splitting csv file")
    parser.add_argument("input_filename", type=str, help="Path to training CSV dataset.")
    parser.add_argument("output_filename", type=str, help="Output filename.")
    parser.add_argument("nr_of_shards", type=int, help="Number of files we want our dataset to be split on.")
    parser.add_argument("--permute", action="store_true", help="Deterministically permutes dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    print(args.input_filename)
    print(args.output_filename)
    print(args.nr_of_shards)
    print(args.permute)
    print(args.seed)

    dirpath = os.path.dirname(args.output_filename)
    # filename = os.path.basename(args.output_filename)
    # print(dirpath)
    # print(filename)
    if dirpath != '':
        os.makedirs(dirpath, exist_ok=True)

    df = pd.read_csv(args.input_filename, header=None)
    if args.permute:
        seed = args.seed
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    k = args.nr_of_shards
    shards = np.array_split(df, k)
    
    for rank, shard in enumerate(shards):
        shard.to_csv(f"{args.output_filename}_{rank}", index=False, header=False)
    return

if __name__ == "__main__":
    main()