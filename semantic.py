# LEVEL 3, TASK 12: SEMANTIC SIMILARITY (NLP) 

# import spacy and load model
import spacy
nlp = spacy.load('en_core_web_md')

# the following code is taken from the task 12 pdf document provided by Hyperiondev
# EXAMPLE 1
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# EXAMPLE 2
tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# EXAMPLE 3
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + "-", similarity) 


# ANSWERS
'''So when observing the comparisons made between the words "cat", "monkey" and "banana", "cat" and
"monkey" are marked as very similar (the number given by the program is: 0.5929930210113525) and this
is most likely because they are both animals. The words "monkey" and "banana" are also marked as very
similar, and this is probably because monkeys eat fruit, and there is also the stereo type that monkeys
love bananas in particular. The words "cat" and "banana", however, are marked as very dissimilar, and
this is because cats arent known to eat fruit - they are carnivores - so the program will not see them
as being very similar. Now, if we were to add another word to the list, such as "lion", the program
will probably mark the similarities between "cat" and "lion"  as being higher than the similarities
than between "cat" and "monkey", because cats and lions are both felines. However, the same will probably
be true for "monkey" and "lion", that their similarities rank higher than that of "cat" and "monkey" because
both lions and monkeys are classed as wild animals, while cats are classed as being domestic animals.'''

'''So for the example.py file we were given for this task, when I changed the model 'en_core_web_md' to
'en_core_web_sm', the results given after running the program were not as precise as those given when
the 'md' model was used. I also got a warning message printed at the top of the terminal stating that
since the 'sm' model doesnt contain 'vectors' its analysis of the sentences given in the example.py file
would not be as accurate as it would have been with the 'md' model - which is what I observed.'''