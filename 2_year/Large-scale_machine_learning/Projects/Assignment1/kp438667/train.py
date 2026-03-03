from mpi4py import MPI
import re
import pandas as pd
import numpy as np
import argparse
import os

# 1. Read dataset.
def get_dataset(filename, rank):
    df = pd.read_csv(f"{filename}_{rank}", header=None)
    assert len(df.shape) == 2
    df.columns = ["text", "label"]
    return df

def dataset_summary(df):
    print("\n--- Head ---\n")
    print(df.head())
    print("\n--- Shape ---\n")
    print(df.shape)
    print("\n--- Datatype ---\n")
    print(df.dtypes)
    print("\n--- Missing values (Nan) ---\n")
    print(df.isna().sum())
    print("\n--- Describe ---\n")
    print(df.describe())

# 2. Text cleaning.
def text_cleaning(text):
    s =  re.sub(r'[^A-Za-z\s]', "", text)
    return s.lower()

def dataset_cleaning(df):
    assert df.columns.to_list() == ["text", "label"]
    assert not df['text'].isna().any()
    df['text'] = df['text'].apply(text_cleaning)
    return df

# 3. Use mpi4py.allreduce method to get global dictionary of number of string occurences in the corpus.
def find_words(texts):
    d = dict()
    for t in texts:
        for w in t.split():
            d[w] = d.get(w,0) + 1
    return d

def allreduce_vocabulary_corpus(local_vocab, comm):
    def merge_dicts(d1, d2, dtype):
        for k,v in d2.items():
            d1[k] = d1.get(k,0) + v
        return d1

    op = MPI.Op.Create(merge_dicts, commute=True)

    global_vocab = comm.allreduce(local_vocab, op=op)
    return global_vocab

# 4. Create local vocabulary.
def dimention_reduction(local_vocab, global_vocab):
    result = set()
    for k in local_vocab.keys():
        if global_vocab[k] > 1:
            result.add(k)
    return sorted(list(result))

# 5. Define gini formulas.
def gini(y, S):
    _, counts = np.unique(y[S], return_counts=True)
    pi = counts / counts.sum()
    return 1 - np.dot(pi,pi) 

def gini_split(y, SL, SR):
    sl = SL.sum()
    sr = SR.sum()
    s = sl+sr
    return sl/s * gini(y, SL) + sr/s * gini(y, SR)

def gain(y, SL, SR, S):
    return gini(y, S) - gini_split(y, SL, SR)

def traverse(X, y, S, features, depth, epsilon=1e-9):
    # TODO: can we vectorize it instead of just using loop with K iterations?
    if depth == 10 or np.abs(gini(y,S)) <= epsilon:
        classes, counts = np.unique(y[S], return_counts=True)
        return [int(classes[np.argmax(counts)])]
    best_gain = 0.
    best_feature=0
    for i in range(len(features)):
        # SL - feature is absent.
        # SR - feature is present.
        SL = S & (~X[:,i])
        SR = S & X[:,i]
        if SL.sum() > 0 and SR.sum() > 0:
            # Gini does not make sense for empty set, so if SL or SR is empty then we set gain to 0.
            current_gain = gain(y, SL, SR, S)
            if current_gain > best_gain + epsilon:
                best_feature = i
                best_gain = current_gain
    if np.abs(best_gain) < epsilon:
        classes, counts = np.unique(y[S], return_counts=True)
        return [int(classes[np.argmax(counts)])]
    SL = S & (~X[:,best_feature])
    SR = S & X[:,best_feature]
    return [str(features[best_feature]), traverse(X, y, SL, features, depth+1, epsilon), traverse(X, y, SR, features, depth+1, epsilon)]

def serializeTree(tree):
    if isinstance(tree, list):
        if len(tree) == 1:
            return "["+str(tree[0])+"]"
        return f'["{tree[0]}", {serializeTree(tree[1])}, {serializeTree(tree[2])}]'
    return str(tree)

def main():
    """
    <dataset_path> <model_output> <n_trees = T> <seed>
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description="Training of Random Forest")
    parser.add_argument("dataset_path", type=str, help="Path to training CSV dataset.")
    parser.add_argument("model_output", type=str, help="Output filename.")
    parser.add_argument("n_trees", type=int, help="Number of trees for the node.")
    parser.add_argument("seed", type=int, help="Seed for deterministic behaviour.")
    args = parser.parse_args()

    # Create folder for output if necessary.
    dirpath = os.path.dirname(args.model_output)
    if dirpath != '':
        os.makedirs(dirpath, exist_ok=True)

    # print(args.dataset_path)
    # print(args.model_output)
    # print(args.n_trees)
    # print(args.seed)

    # Get MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    df = get_dataset(args.dataset_path, rank)
    # print(df.shape)
    df = dataset_cleaning(df)
    # print("After cleaning head:", df.head())
    texts = df['text'].to_list()
    local_dict = find_words(texts)
    # print(list(local_dict.items())[:5])
    # for i, (k,v) in enumerate(local_dict.items()):
    #     print(f"{i}th pair: ({k}, {v})")
    #     if i == 4:
    #         break
    global_dict = allreduce_vocabulary_corpus(local_dict, comm)
    # print(list(local_dict.items())[:5])
    # print(list(global_dict.items())[:5])

    vocab = np.array(dimention_reduction(local_dict, global_dict))
    # print(vocab[:10])
    # print(sorted(list(local_dict.keys())) [:10])

    T = args.n_trees
    V = len(vocab)
    K = np.sqrt(np.array([V])).astype(int).item()
    # print(T, V, K)
    # print("table size:", V*df.shape[0])
    # print("smaller version:", K*df.shape[0])
    # We can see that it is impossible to keep whole table in memory. 
    # Instead for every node (max= +-2^10), chosen feature (K=sqrt(V) features and observation we check if feature is in observation.
    # So complexity = O(nodes*sqrt(V)*N) (probably much less because after next node has smaller set of observations). 

    with open(f'{args.model_output}_{rank}', 'w') as output:
        output.write(" ".join(vocab.tolist()) + "\n")
        for i in range(T):
            np.random.seed(args.seed*1000*rank+i)
            # Boostrap Sampling
            n = df.shape[0]
            indices = np.random.randint(n, size=n)
            sample = df.iloc[indices]
            # Choose K features
            features = np.random.choice(vocab, K, replace=False)
            X = np.zeros((n, K), dtype=bool)
            for i, text in enumerate(sample['text']):
                words = set(text.split())
                for j, feat in enumerate(features):
                    if feat in words:
                        X[i,j] = True
            y = sample['label'].values
            output.write(serializeTree(traverse(X, y, np.ones(n, dtype=bool), features, 1)) + "\n")
    return

if __name__ == "__main__":
    main()