# HMM Viterbi PoS-Tagging

PoS-Tagging using Hidden Markov Model (HMM) with Viterbi Algorithm.

## Dependencies

```bash
pip install numpy tqdm
```

## Usage

Inside the `code` folder:

```bash
python main.py
```

## Results

Check https://nlpprogress.com/english/part-of-speech_tagging.html to compare.

### Mac-Morpho

Dataset: http://nilc.icmc.usp.br/macmorpho/

Accuracy training with the training set (macmorpho-train.txt) and testing on the test set (macmorpho-test.txt): 92.18%

## WSJ
Dataset: https://aclanthology.org/J93-2004/

Accuracy training with the training set (WSJ_02-21.txt) and testing on the test set (WSJ_24.txt): 94.18%