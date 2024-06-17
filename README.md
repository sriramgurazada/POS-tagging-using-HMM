Developed a part-of-speech tagging system using Hidden Markov Models (HMM) on the Wall Street Journal section of the Penn Treebank:
Vocabulary Creation: Generated a vocabulary from training data, handling unknown words by replacing rare occurrences with a special token <unk>, and outputting the vocabulary to a vocab.txt file.
HMM Model Training: Learned emission and transition parameters from training data, saving the model parameters in a JSON file (hmm.json).
Greedy and Viterbi Decoding: Implemented and evaluated both greedy and Viterbi decoding algorithms for part-of-speech tagging, reporting accuracy on development data and generating predictions for test data.
