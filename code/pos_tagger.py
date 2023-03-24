import numpy as np
from tqdm import tqdm


def preprocess_to_infer(text, vocab_idx, lower_vocab):
    text = text.strip().split(" ")

    if len(text) == 0:
        return ""

    input_text = []
    for token in text:
        if token in vocab_idx:
            input_text.append(token)
        elif token.lower() in lower_vocab:
            input_text.append(lower_vocab[token.lower()])
        else:
            input_text.append("__unknown__")

    return text, input_text


def load_dataset(filepath, vocab_idx=None, lower_vocab=None):
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = f.readlines()

    dataset = [sentence.strip() for sentence in dataset]
    dataset = [sentence.split(" ") for sentence in dataset if len(sentence) > 0]
    dataset = [[tuple(token.split("_")) for token in sentence] for sentence in dataset]

    if vocab_idx is not None:
        # if token not in vocab, make the token __unknown__
        # unless a lower mapping version is in vocab, if so, use the lower mapping version

        # (original token, mapped token to vocab, tag)
        dataset = [
            [(token, token, tag) for token, tag in sentence] for sentence in dataset
        ]

        for i in range(len(dataset)):
            for j in range(len(dataset[i])):
                token, _, tag = dataset[i][j]
                if token not in vocab_idx:
                    if token.lower() in lower_vocab:
                        dataset[i][j] = (token, lower_vocab[token.lower()], tag)
                    else:
                        dataset[i][j] = (token, "__unknown__", tag)

    return dataset


def get_matrices(dataset: list[tuple[str, str]]):
    vocab, tagset = set(), set()
    for sentence in dataset:
        for token, tag in sentence:
            vocab.add(token)
            tagset.add(tag)

    vocab = sorted(vocab) + ["__unknown__"]
    vocab_idx = {token: i for i, token in enumerate(vocab)}
    lower_vocab = {token.lower(): token for token in vocab_idx.keys()}

    tagset = sorted(tagset)
    tagset_idx = {tag: i for i, tag in enumerate(tagset)}

    V = len(vocab)
    T = len(tagset)

    # Transition matrix has one more tag for source: __start__
    # and one more for target: __end__
    transition_matrix = np.zeros((T + 1, T + 1))
    emission_matrix = np.zeros((T, V))

    # Computing:
    # transition_matrix: P(curr_tag | prev_tag)
    # emission_matrix: P(word | curr_tag)
    for sentence in tqdm(dataset, desc="Computing probs..."):
        assert len(sentence) > 0
        prev_tag_idx = T  # T is the index for __start__
        for token, tag in sentence:
            curr_tag_idx = tagset_idx[tag]
            transition_matrix[prev_tag_idx, curr_tag_idx] += 1
            prev_tag_idx = curr_tag_idx

            emission_matrix[curr_tag_idx, vocab_idx[token]] += 1
        transition_matrix[prev_tag_idx, T] += 1  # T is the index for __end__

    # Laplacian Smoothing
    epsilon = 0.001
    transition_matrix += epsilon
    emission_matrix += epsilon

    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
    emission_matrix /= np.sum(emission_matrix, axis=1, keepdims=True)

    # Computing log of prob, to avoid tiny numbers multiplications
    # For now on, the multiplications will be replaced by sum (log property)
    transition_matrix = np.log(transition_matrix)
    emission_matrix = np.log(emission_matrix)

    return transition_matrix, emission_matrix, tagset, vocab_idx, lower_vocab


def inference(text, transition_matrix, emission_matrix, tagset, vocab_idx):
    T = emission_matrix.shape[0]
    # TODO I think it is not needed to init with -Inf since we have the epsilon now
    # TODO Also, do not need to record every step, only keep the last best_probs column
    best_probs = np.zeros((T, len(text))) - np.inf
    best_paths = np.zeros((T, len(text)), dtype=np.int64)

    # initialize
    k = T  # __start__ tag index in transition matrix
    wi = 0
    vwi = vocab_idx[text[wi]]

    # for ti in range(T):
    #     best_probs[ti, wi] = transition_matrix[k, ti] + emission_matrix[ti, vwi]
    #     best_paths[ti, wi] = k
    # For reference, above can be replaced with below

    # probability of comming from __start__ tag to every tag and emit the word #0
    # transition_matrix -1 to exclude __end__ tag
    best_probs[:, wi] = transition_matrix[k, :-1] + emission_matrix[:, vwi]
    best_paths[:, wi] = k

    # forward
    idxs_t = np.arange(T)
    for wi in range(1, len(text)):
        vwi = vocab_idx[text[wi]]
        # transition_matrix is using -1 to exclude the __start__ row and __end__ column
        prev_probs = best_probs[:, [wi - 1]] + transition_matrix[:-1, :-1]
        ks = np.argmax(prev_probs, axis=0)

        # for ti in range(T):
        #     k = ks[ti]
        #     best_probs[ti, wi] = prev_probs[k, ti] * emission_matrix[ti, vwi]
        # For reference, above can be replaced with below
        best_probs[:, wi] = prev_probs[ks, idxs_t] + emission_matrix[idxs_t, vwi]
        best_paths[:, wi] = ks

    # backtrack
    result = []
    k = np.argmax(best_probs[:, len(text) - 1])
    result.append(tagset[k])
    for wi in range(len(text) - 2, -1, -1):
        k = best_paths[k, wi + 1]
        result.append(tagset[k])

    result = result[::-1]

    return result


def evaluate(dataset, transition_matrix, emission_matrix, tagset, vocab_idx):
    acc = 0
    count_tags = 0
    for sentence in tqdm(dataset, desc="Infering sentences.."):
        _, mapped_text, tags = zip(*sentence)
        Y_hat = inference(
            mapped_text, transition_matrix, emission_matrix, tagset, vocab_idx
        )
        count_tags += len(tags)
        acc += sum([y == y_hat for y, y_hat in zip(tags, Y_hat)])

    acc /= count_tags

    return acc
