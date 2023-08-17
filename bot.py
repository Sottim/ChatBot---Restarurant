#This is the main file of our chatBot 

import json
import torch 
import random

from tools import all_words
from samples import create_data, create_vector
from train import train_fn

f = open("all_intents.json")
main_dt = json.load(f)

#Vocabulary
words_list = sorted(all_words(main_dt))

intents_mapping = {}
intents_response = {} #For the reverse of mappings

#Done to create the mapping of intent in the given data with the index
for i, d in enumerate(main_dt["data"]):
    intents_mapping[d["intent"]] = i
    intents_response[i] = d["responses"]

#For the reverse mapping of the above intents_map
reverse_mappings = {}
for key, value in intents_mapping.items():
    reverse_mappings[value] = key

#Creation of training data
train, target = create_data(main_dt, words_list, intents_mapping)
model = train_fn(train, target, len(words_list), 6)

#Chatting with the network
while True:
    input = input("Enter your query: ")
    vec = create_vector(input, words_list)
    input_vector = torch.as_tensor(vec, dtype = torch.float32)
    output = model(input_vector)

    pred_num = output.argmax().item()
    bot_response = intents_response(pred_num)

    print("Bot: ", bot_response[random.randint(0,2)])
    
