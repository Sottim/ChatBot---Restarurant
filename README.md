# ChatBot For Restaurant
This is a simple chatbot using simple Neural Network.

## Algorithm Implementation :

1. Create the vocubulary after removing the punctuations and Stemming

The vocabulary is created from the intents file. We will iterate through all the queries and then remove punctuations and lastly perform stemming. For the stemming part, we will be using the PorterStemmer module from nltk.

2. Generate training data by creating vectiors using Bag of Words.
Here we create data using the bag of words algorithm. We will also define two variable which are intents_mapping and reverse_intents mapping for changing the target into numbers. In our implementation, all the queries from all 6 elements will be converted to a 48 dimensional vector which will be fed to the neural network for prediction.

3. Define the model and training of the data
We define a two layer simple neural network using relu as the activation function. The input, output and the hidden dimensions are kept as 48 , 6 , 16. 

4. Run inferences on the chatbot using the messages that the chatbot has not seen

## Input Example : 

words_list or Voculabry contains all the words after i.e words_list = [hi, when, how, you, .....around 48 of them]

input_sent = "Hi, how are you ?

word_list = [hi, how, are, you] //After removing punctuations and stemming.

Input vector = [1, 0, 1, 1, 0, 0, 1, 0, 0] //Based on wheather the word from words_list is present in word_list or not.

This input vector is for the input to the Network and is from the user side and it has not been seen by the network.


