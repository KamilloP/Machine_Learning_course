from mpi4py import MPI
import re
import pandas as pd
import numpy as np
import argparse
import ast
import os


def text_cleaning(text):
    s =  re.sub(r'[^A-Za-z\s]', "", text)
    return s.lower()

def traverse(tree, q_set):
    assert len(tree) in {1,3}
    if len(tree) == 1:
        return tree[0]
    assert isinstance(tree[0], str)
    if tree[0] in q_set:
        return traverse(tree[2], q_set)
    return traverse(tree[1], q_set)

def main():
    """
    <model_input> <query_input> <predictions_output>
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description="Classifying with Random Forest")
    parser.add_argument("model_input", type=str, help="Path to model.")
    parser.add_argument("query_input", type=str, help="Path to query input file.")
    parser.add_argument("predictions_output", type=str, help="Desired output filename (with path).")
    args = parser.parse_args()

    # Create folder for output if necessary.
    dirpath = os.path.dirname(args.predictions_output)
    if dirpath != '':
        os.makedirs(dirpath, exist_ok=True)

    # Get MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Model:
    with open(f'{args.model_input}_{rank}') as model:
        vocab = model.readline()
        vocab = set(vocab.split(" ")) # Now it is set.
        trees = []
        for x in model:
            trees.append(ast.literal_eval(x))
            assert isinstance(trees[-1], list), f"Expected list, but got {type(trees[-1])}"
    
    # Query:
    results = []
    with open(args.query_input) as query:
        for q in query:
            q_set = vocab.intersection(set(text_cleaning(q).split()))
            answers = []
            for t in trees:
                answers.append(traverse(t, q_set))
            results.append(answers)
    # Gather all votes:
    data = comm.gather(results, root=0)

    # Choose majority voted class and write it to file:
    if rank == 0:
        data = np.array(data)
        # data.shape = (nr_of_estimators, nr_of_queries, nr_of_trees)
        E, Q, T = data.shape
        data = data.transpose(1, 0, 2).reshape(Q, E*T)
        with open(args.predictions_output, 'w') as output:
            for i in range(Q):
                classes, counts = np.unique(data[i], return_counts=True)
                # https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html: '# Only the first occurrence is returned.'
                output.write(str(classes[np.argmax(counts)]) + "\n")
    else:
        assert data == None
    return

if __name__ == "__main__":
    main()