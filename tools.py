import re
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = []

#Removing Punctuations:
# Given Sentence : Hey ! What's your name ? 
# After removing punctuations : Hey whats your name

def remove_punctuations(sent):
    return re.sub(r'[^\w\s]', "", sent)

# Stemming
# Given Sentence : "What are your working hours" 
# Stemmed : "what are your work hour"
def stemming(word_list):
    return [ps.stem(w) for w in word_list]

def all_words(main_dt):
    for d in main_dt["data"]:
        for q in d["query"]:
            
            #we get the clean queries without any punctuations
            clean_query = remove_punctuations(q)
            #each query to have unique elements of list
            word_list = clean_query.lower().split()
            #Pass the list of words to stemming function
            stemming_list = stemming(word_list)
            words.extend(stemming_list)
    # set: prevents duplicate elements 
    # and pass it as a list so that we can take care of the list manipulation later on    
    return list(set(words))
