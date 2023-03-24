import os
from pos_tagger import load_dataset, get_matrices, evaluate


def main():
    train_filepath = os.path.join("..", "data", "WSJ", "WSJ_02-21.txt")
    test_filepath = os.path.join("..", "data", "WSJ", "WSJ_24.txt")

    # train_filepath = os.path.join("..", "data", "macmorpho", "macmorpho-train.txt")
    # test_filepath = os.path.join("..", "data", "macmorpho", "macmorpho-test.txt")

    dataset = load_dataset(train_filepath)
    transition_matrix, emission_matrix, tagset, vocab_idx, lower_vocab = get_matrices(
        dataset
    )

    # to load test or dev datasets, you must pass vocab_idx (acquired from training data) to the function
    # or it will not work properly
    dataset = load_dataset(test_filepath, vocab_idx, lower_vocab)
    acc = evaluate(
        dataset,
        transition_matrix,
        emission_matrix,
        tagset,
        vocab_idx,
    )
    print(f"{train_filepath} -> {test_filepath}: {acc}")


if __name__ == "__main__":
    main()
