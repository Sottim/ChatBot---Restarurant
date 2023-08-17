from tools import remove_punctuations
from tools import stemming
import torch


train = []
target = []

#Function for the creation of vector 

def create_vector(sent, words_list):
    vector = []
    clean_sentence = remove_punctuations(sent)
    words_list = clean_sentence.lower().split()

    stemming_list = stemming(words_list)
    
    #Creation of vector depending upon the if the word is available or not in the word_list
    for w in words_list:
        if w in stemming_list:
            vector.append(1)
        else:
            vector.append(0)
    return vector

# Function defined for the data creation
def create_data(main_dt, words_list, intents_mapping):

    max_sequence_length = 0  # Track the maximum sequence length

    for d in main_dt["data"]:
        local_intent = d["intent"]
        queries = d["query"]

        for q in queries:
            vector = create_vector(q, words_list)
            train.append(vector)
            target.append(intents_mapping[local_intent])

            # Update the maximum sequence length
            max_sequence_length = max(max_sequence_length, len(vector))

    # Apply padding to the sequences
    padded_train = []
    for sequence in train:
        # Pad the sequence with zeros
        padded_sequence = sequence + [0] * (max_sequence_length - len(sequence))
        padded_train.append(padded_sequence)

    # Convert train and target to tensors
    train_tensor = torch.tensor(padded_train, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.int64)

    return train_tensor, target_tensor


    print(len(train))
    print(len(train[8]))
    print(len(target))
    # return train, target



