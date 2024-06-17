#!/usr/bin/env python
# coding: utf-8

# - Name : Devi Venkata Sai Sriram Chandra
# - ID.No: 5989499046

# ### Task 1: Vocabulary Creation 

# In[1]:


def load_and_preprocess_data(filename, threshold=3):
    # Initialize a dictionary to count occurrences of each word
    word_counts = {}
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                _, word, _ = line.strip().split('\t')
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
    
    # Replace rare words with <unk>
    vocab = {'<unk>': 0}
    for word, count in word_counts.items():
        if count < threshold:
            vocab['<unk>'] += count
        else:
            vocab[word] = count
            
    return vocab


# In[2]:


def create_and_export_vocab(vocab, filename='vocab.txt'):
    # Ensure <unk> is first by removing it from the vocab, then adding it back at the beginning
    unk_count = vocab.pop('<unk>', 0)
    sorted_vocab_items = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    
    # Prepend <unk> to the sorted list
    sorted_vocab = [('<unk>', unk_count)] + sorted_vocab_items
    
    # Write the vocab to a file with <unk> at index 0
    with open(filename, 'w', encoding='utf-8') as file:
        for index, (word, count) in enumerate(sorted_vocab):
            file.write(f"{word}\t{index}\t{count}\n")
    
    # Return the count of <unk> for further use
    return unk_count


# In[3]:


train_data_filename = 'data/train'  # Make sure this path is correct

# Load and preprocess the training data
vocab = load_and_preprocess_data(train_data_filename)

# Create and export the vocabulary, capturing the <unk> count
unk_count = create_and_export_vocab(vocab)

# Output the total size of the vocabulary (now without <unk>) and occurrences of <unk>
print(f"Total size of vocabulary: {len(vocab) + 1}")  # Adding 1 for <unk>
print(f"Total occurrences of '<unk>': {unk_count}")


# In[ ]:





# The above code processes textual data to build a vocabulary, counting occurrences of each word and replacing rare words with "&lt;unk&gt;". It then sorts this vocabulary by frequency and exports it to a file, detailing each word's rank and occurrence count. The main process involves loading data from a specified file, preprocessing it based on a frequency threshold, and creating an exportable vocabulary list. Finally, it outputs the total vocabulary size and occurrences of "&lt;unk&gt;".

# ### Task 2: HMM probabilities

# In[4]:


def load_training_data_and_collect_counts(filename):
    # Initialize dictionaries for counts
    transition_counts = {}
    emission_counts = {}
    state_counts = {}
    prev_state = None  # Track the previous state for transition counts
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Process lines within sentences
                _, word, state = line.strip().split('\t')
                
                # Update emission counts
                emission_counts[(state, word)] = emission_counts.get((state, word), 0) + 1
                
                # Update state counts
                state_counts[state] = state_counts.get(state, 0) + 1
                
                # Update transition counts if not the first word in a sentence
                if prev_state is not None:
                    transition_counts[(prev_state, state)] = transition_counts.get((prev_state, state), 0) + 1
                
                prev_state = state
            else:
                prev_state = None  # Reset for the next sentence
                
    return transition_counts, emission_counts, state_counts


# In[5]:


def calculate_probabilities(transition_counts, emission_counts, state_counts):
    transition_probs = {k: v / state_counts[k[0]] for k, v in transition_counts.items()}
    emission_probs = {k: v / state_counts[k[0]] for k, v in emission_counts.items()}
    return transition_probs, emission_probs


# In[6]:


import json
     
def export_model_to_json(transition_probs, emission_probs, filename='hmm.json'):
    # Convert tuple keys to string
    transition_probs_str = {f'{k[0]}->{k[1]}': v for k, v in transition_probs.items()}
    emission_probs_str = {f'{k[0]}->{k[1]}': v for k, v in emission_probs.items()}
    
    model = {
        'transition': transition_probs_str,
        'emission': emission_probs_str,
        'state_counts': state_counts
    }
    
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(model, file, ensure_ascii=False, indent=4)


# In[7]:


# Define the path to your training data
train_data_filename = 'data/train'

# Load training data and collect counts
transition_counts, emission_counts, state_counts = load_training_data_and_collect_counts(train_data_filename)

# Calculate probabilities
transition_probs, emission_probs = calculate_probabilities(transition_counts, emission_counts, state_counts)

# Export the model
export_model_to_json(transition_probs, emission_probs)

# Output the number of transition and emission parameters
print(f"Number of transition parameters: {len(transition_probs)}")
print(f"Number of emission parameters: {len(emission_probs)}")


# The above code reads training data to compute transition and emission counts for a Hidden Markov Model (HMM), including counts of states. It then calculates the transition and emission probabilities based on these counts. The probabilities are formatted into a JSON model, exporting transition and emission probabilities along with state counts. Finally, it outputs the total number of transition and emission parameters calculated from the training data.

# ### Task 3: Greedy Decoding with HMM

# In[8]:


def load_hmm_model(model_filename):
    with open(model_filename, 'r') as file:
        model = json.load(file)
    transition = model['transition']
    emission = model['emission']
    
    # Extract states from emission parameters
    states = set(key.split('->')[0] for key in emission.keys())
    
    # Assuming uniform start probabilities (adjust as needed)
    start_prob = {state: 1/len(states) for state in states}
    
    return states, start_prob, transition, emission


# In[9]:


def predict_tags(sentence, transition, emission):
    predicted_tags = []
    for word in sentence:
        max_prob = -1
        best_tag = None
        for state, prob in emission.items():
            tag, emit_word = state.split('->')
            if emit_word == word and prob > max_prob:
                max_prob = prob
                best_tag = tag
        
        # Enhanced handling for unknown words
        if best_tag is None:
            if word.endswith('ed'):
                best_tag = 'VBD'  # Past-tense verb
            elif word[0].isupper():
                best_tag = 'NNP'  # Proper noun
            else:
                best_tag = 'NN'  # Default to common noun

        predicted_tags.append(best_tag)
    return predicted_tags


# In[10]:


def greedy_decoding(dev_filename, transition, emission, output_filename):
    with open(dev_filename, 'r') as dev_file, open(output_filename, 'w') as out_file:
        sentence = []
        for line in dev_file:
            if line.strip():
                index, word, _ = line.strip().split()
                sentence.append(word)
            else:
                # Predict tags for the sentence
                predicted_tags = predict_tags(sentence, transition, emission)
                for i, (word, tag) in enumerate(zip(sentence, predicted_tags), 1):
                    out_file.write(f"{i}\t{word}\t{tag}\n")
                out_file.write("\n")  # New line for sentence separation
                sentence = []  # Reset sentence



# In[11]:


# Load HMM model
states, start_prob, transition, emission = load_hmm_model('hmm.json')


# In[12]:


# Predict and output tags for development data
greedy_decoding('data/dev', transition, emission, 'greedy_dev.out')


# In[13]:


get_ipython().system('python eval.py -p greedy_dev.out -g data/dev')


# In[14]:


def test_greedy_decoding(dev_filename, transition, emission, output_filename):
    with open(dev_filename, 'r') as dev_file, open(output_filename, 'w') as out_file:
        sentence = []
        for line in dev_file:
            if line.strip():
                index, word = line.strip().split()
                sentence.append(word)
            else:
                # Predict tags for the sentence
                predicted_tags = predict_tags(sentence, transition, emission)
                for i, (word, tag) in enumerate(zip(sentence, predicted_tags), 1):
                    out_file.write(f"{i}\t{word}\t{tag}\n")
                out_file.write("\n")  # New line for sentence separation
                sentence = []  # Reset sentence



# In[15]:


# Predict and output tags for development data
test_greedy_decoding('data/test', transition, emission, 'greedy.out')


# Greedy decoding in the context of Hidden Markov Models (HMMs) involves choosing the most likely state (or tag) for each observation in a sequence independently of the others, based on the highest emission probability at each step. Unlike Viterbi decoding, which considers the entire sequence to find the optimal path, greedy decoding makes a local, immediate choice at each step without regard to future consequences. This approach is simpler and faster but often less accurate than methods that consider the entire sequence, such as the Viterbi algorithm.

# The code defines functions to load a Hidden Markov Model (HMM) from a JSON file, predict part-of-speech tags for sentences using the model, and perform greedy decoding on a dataset to predict and output tags. It handles unknown words by assigning default tags based on word features (e.g., past-tense verbs, proper nouns). The predict_tags function iterates through each word in a sentence, using emission probabilities to choose the most likely tag, with special rules for unknown words. Finally, the model is applied to development and test datasets to generate tagged outputs, showcasing practical use of the HMM for sequence tagging tasks.

# ### Task 4: Viterbi decoding using HMM

# In[16]:


import json

def load_hmm_model(model_filename):
    with open(model_filename, 'r') as file:
        model = json.load(file)
    transition = model['transition']
    emission = model['emission']
    states = set(key.split('->')[0] for key in emission.keys())
    start_prob = {state: 1/len(states) for state in states}  # Uniform start probabilities
    return states, start_prob, transition, emission


# In[17]:


def viterbi_decode(sentence, states, start_prob, transition, emission):
    # Viterbi and path matrices
    viterbi = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for state in states:
        viterbi[0][state] = start_prob.get(state, 1e-8) * emission.get(f"{state}->{sentence[0]}", 1e-8)
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(sentence)):
        viterbi.append({})
        new_path = {}

        for curr_state in states:
            (max_prob, max_state) = max(
                (viterbi[t-1][prev_state] *
                 transition.get(f"{prev_state}->{curr_state}", 1e-8) *
                 emission.get(f"{curr_state}->{sentence[t]}", 1e-8), prev_state)
                for prev_state in states
            )

            viterbi[t][curr_state] = max_prob
            new_path[curr_state] = path[max_state] + [curr_state]

        path = new_path

    # Find the final state with maximum probability
    n = len(sentence) - 1
    (max_prob, max_state) = max((viterbi[n][state], state) for state in states)

    return path[max_state]


# In[18]:


def viterbi_predict_and_write(dev_filename, states, start_prob, transition, emission, output_filename):
    with open(dev_filename, 'r') as dev_file, open(output_filename, 'w') as out_file:
        sentence = []
        for line in dev_file:
            if line.strip():
                _, word, _ = line.strip().split('\t')
                sentence.append(word)
            else:
                if sentence:  # Check if the sentence is not empty
                    predicted_tags = viterbi_decode(sentence, states, start_prob, transition, emission)
                    for i, (word, tag) in enumerate(zip(sentence, predicted_tags), 1):
                        out_file.write(f"{i}\t{word}\t{tag}\n")
                    out_file.write("\n")  # Sentence separation
                sentence = []  # Reset for next sentence
        if sentence:  # Handle last sentence if file doesn't end with a newline
            predicted_tags = viterbi_decode(sentence, states, start_prob, transition, emission)
            for i, (word, tag) in enumerate(zip(sentence, predicted_tags), 1):
                out_file.write(f"{i}\t{word}\t{tag}\n")
            out_file.write("\n")


# In[19]:


states, start_prob, transition, emission = load_hmm_model('hmm.json')
viterbi_predict_and_write('data/dev', states, start_prob, transition, emission, 'viterbi_dev.out')


# In[20]:


get_ipython().system('python eval.py -p viterbi_dev.out -g data/dev')


# In[21]:


def test_viterbi_predict_and_write(dev_filename, states, start_prob, transition, emission, output_filename):
    with open(dev_filename, 'r') as dev_file, open(output_filename, 'w') as out_file:
        sentence = []
        for line in dev_file:
            if line.strip():
                _, word = line.strip().split('\t')
                sentence.append(word)
            else:
                if sentence:  # Check if the sentence is not empty
                    predicted_tags = viterbi_decode(sentence, states, start_prob, transition, emission)
                    for i, (word, tag) in enumerate(zip(sentence, predicted_tags), 1):
                        out_file.write(f"{i}\t{word}\t{tag}\n")
                    out_file.write("\n")  # Sentence separation
                sentence = []  # Reset for next sentence
        if sentence:  # Handle last sentence if file doesn't end with a newline
            predicted_tags = viterbi_decode(sentence, states, start_prob, transition, emission)
            for i, (word, tag) in enumerate(zip(sentence, predicted_tags), 1):
                out_file.write(f"{i}\t{word}\t{tag}\n")
            out_file.write("\n")


# In[22]:


states, start_prob, transition, emission = load_hmm_model('hmm.json')
test_viterbi_predict_and_write('data/test', states, start_prob, transition, emission, 'viterbi.out')


# Viterbi decoding is an algorithm used with Hidden Markov Models (HMMs) to find the most likely sequence of hidden states based on a sequence of observed events. It uses dynamic programming to efficiently compute the probabilities of state sequences, selecting the path that maximizes these probabilities. This method is particularly useful in applications such as speech recognition, part-of-speech tagging, and bioinformatics for decoding the underlying state sequences from observed data.

# The above code defines a function to load a Hidden Markov Model (HMM) from a JSON file and another function to decode sentences using the Viterbi algorithm, given the model's states, start probabilities, transition probabilities, and emission probabilities. The Viterbi function calculates the most likely sequence of states (tags) for a given sentence by maximizing the probabilities of paths through the states based on the observed sequence. It initializes probabilities for the first word based on start probabilities and emission probabilities, then iterates through the sentence, updating probabilities and paths for reaching each state. The final output is the path with the highest probability for the entire sentence.

# #### Report:

# - **Selected threshold for unknown words replacement :** 3
# - **Total size of vocabulary:** 16920
# - **Total occurrences of '&lt;unk&gt;':** 32537
# - **Number of transition parameters:** 1351
# - **Number of emission parameters:** 50286
# - **Accuracy of Greedy algorithm on Dev set :** 90.18%
# - **Accuracy of viterbi algorithm on Dev set :** 94.84%
# 
# 

# In[ ]:




