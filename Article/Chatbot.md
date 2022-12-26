<style>
a:link {
  text-decoration:none;
}
blockquote{
  padding: 0.5em;
  background: #ddeaff;
  border-left: 8px solid #0072b3;
}
</style>

<base target="_blank">

## Build a Simple Chatbot in Python

A chatbot (conversational interface, AI agent) is a computer program that can understand human language and converse with a user via a website or a messaging app. Chatbots can handle various tasks online - from answering simple questions and scheduling calls to gathering customer feedback.

History of chatbots dates back to 1966 when a computer program called ELIZA was invented by Weizenbaum. It imitated the language of a psychotherapist from only 200 lines of code. You can still converse with it here: [Eliza](http://psych.fullerton.edu/mbirnbaum/psych101/Eliza.htm?utm_source=ubisend.com&utm_medium=blog-link&utm_campaign=ubisend). 

### Import necessary libraries
```python
import io
import numpy as np
import nltk # NLTK is a leading platform for building Python programs to work with human language data.
import string # provides the ability to do complex variable substitutions and value formatting
import random
import warnings
warnings.filterwarnings('ignore')
```
### TfidfVectorizer
The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features. FastText and Word2Vec Word Embeddings Python Implementation.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
```

### Downloading and installing NLTK
NLTK(Natural Language Toolkit) is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries.

[Natural Language Processing with Python](http://www.nltk.org/book/) provides a practical introduction to programming for language processing.

```python
pip install nltk

# Output
Requirement already satisfied: nltk in c:\users\username\anaconda3\lib\site-packages (3.7)
Requirement already satisfied: click in c:\users\username\anaconda3\lib\site-packages (from nltk) (8.0.4)
Requirement already satisfied: joblib in c:\users\username\anaconda3\lib\site-packages (from nltk) (1.1.0)
Requirement already satisfied: regex>=2021.8.3 in c:\users\username\anaconda3\lib\site-packages (from nltk) (2022.3.15)
Requirement already satisfied: tqdm in c:\users\username\anaconda3\lib\site-packages (from nltk) (4.64.0)
Requirement already satisfied: colorama in c:\username\saurabh\anaconda3\lib\site-packages (from click->nltk) (0.4.4)
Note: you may need to restart the kernel to use updated packages.
```


```python
from nltk.stem import WordNetLemmatizer
nltk.download('punkt') # Using the Punkt tokenizer
nltk.download('wordnet') # Using the WordNet dictionary
nltk.download('popular', quiet=True) # for downloading packages

# Output

[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\username\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\username\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
Out[4] True
```

### Reading in the corpus
For our example,we will be using the Wikipedia page for chatbots as our corpus. Copy the contents from the page and place it in a text file named ‘chatbot.txt’. However, you can use any corpus of your choice.

```python
f = open('chatbot.txt','r',errors = 'ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower() # Converts text lowercase
```

The main issue with text data is that it is all in text format (strings). However, the Machine learning algorithms need some sort of numerical feature vector in order to perform the task. So before we start with any NLP project we need to pre-process it to make it ideal for working. Basic text pre-processing includes:

- Converting the entire text into **uppercase** or **lowercase**, so that the algorithm does not treat the same words in different cases as different
- **Tokenization:** Tokenization is just the term used to describe the process of converting the normal text strings into a list of tokens i.e words that we actually want. Sentence tokenizer can be used to find the list of sentences and Word tokenizer can be used to find the list of words in strings.

The NLTK data package includes a pre-trained Punkt tokenizer for English.

- Removing **Noise** i.e everything that isn’t in a standard number or letter.
- Removing the **Stop words**. Sometimes, some extremely common words which would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called stop words
- Stemming: **Stemming** is the process of reducing inflected (or sometimes derived) words to their stem, base or root form — generally a written word form. Example if we were to stem the following words: “Stems”, “Stemming”, “Stemmed”, “and Stemtization”, the result would be a single word “stem”.
- **Lemmatization:** A slight variant of stemming is lemmatization. The major difference between these is, that, stemming can often create non-existent words, whereas lemmas are actual words. So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma. Examples of Lemmatization are that “run” is a base form for words like “running” or “ran” or that the word “better” and “good” are in the same lemma so they are considered the same.

### Tokenisation

```python
sent_tokens = nltk.sent_tokenize(raw_doc) # Converts doc to list of sentences
word_tokens = nltk.word_tokenize(raw_doc) # Converts doc to list of words
```

```python
sent_tokens[:2]

# Output

Out[7] ['a chatbot (also known as a talkbot, chatterbot, bot, im bot, interactive agent, or artificial conversational entity) is a computer program or an artificial intelligence which conducts a conversation via auditory or textual methods.',
 'such programs are often designed to convincingly simulate how a human would behave as a conversational partner, thereby passing the turing test.']
```

```python
word_tokens[:2]

# Output

Out[8] ['a', 'chatbot']
```

### Preprocessing
We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.

```python
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
```

### Keyword matching
Next, we shall define a function for a greeting by the bot i.e if a user’s input is a greeting, the bot shall return a greeting response.ELIZA uses a simple keyword matching for greetings. We will utilize the same concept here.
```python
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
```

### Generating Response
#### Bag of Words
After the initial preprocessing phase, we need to transform text into a meaningful vector (or array) of numbers. The bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:

- A vocabulary of known words.
- A measure of the presence of known words.

Why is it is called a “bag” of words? That is because any information about the order or structure of words in the document is discarded and the model is only **concerned with whether the known words occur in the document, not where they occur in the document.**

The intuition behind the Bag of Words is that documents are similar if they have similar content. Also, we can learn something about the meaning of the document from its content alone.

For example, if our dictionary contains the words {Learning, is, the, not, great}, and we want to vectorize the text “Learning is great”, we would have the following vector: (1, 1, 0, 0, 1).

### TF-IDF Approach
A problem with the Bag of Words approach is that highly frequent words start to dominate in the document (e.g. larger score), but may not contain as much “informational content”. Also, it will give more weight to longer documents than shorter documents.

One approach is to rescale the frequency of words by how often they appear in all documents so that the scores for frequent words like “the” that are also frequent across all documents are penalized. This approach to scoring is called Term Frequency-Inverse Document Frequency, or TF-IDF for short, where:

#### Term Frequency: is a scoring of the frequency of the word in the current document.
&nbsp; &nbsp; TF = (Number of times term t appears in a document)/(Number of terms in the document)

#### Inverse Document Frequency: is a scoring of how rare the word is across documents.

&nbsp; &nbsp; IDF = 1+log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.

### Cosine Similarity

Tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus

&nbsp; &nbsp; Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||

where d1,d2 are two non zero vectors.

To generate a response from our bot for input questions, the concept of document similarity will be used. We define a function response which searches the user’s utterance for one or more known keywords and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, it returns a response:” I am sorry! I don’t understand you”

```python
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
```

Finally, we will feed the lines that we want our bot to say while starting and ending a conversation depending upon user’s input.


```python
flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!\nTopics you can ask: Chatbot/talkbot, Eliza, Artificial Intelligence, Development, Toys, Design, Analytics, APIs")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")

# Output

ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!
Topics you can ask: Chatbot/talkbot, Eliza, Artificial Intelligence, Development, Toys, Design, Analytics, APIs
Hi
ROBO: hello
Chatbot
ROBO: design
the chatbot design is the process that defines the interaction between the user and the chatbot.the chatbot designer will define the chatbot personality, the questions that will be asked to the users, and the overall interaction.it can be viewed as a subset of the conversational design.
Eliza
ROBO: history of chatbots dates back to 1966 when a computer program called eliza was invented by weizenbaum.
APIs
ROBO: apis
there are many apis available for building your own chatbots, such as aarc.
Analytics
ROBO: it can be divided into design, building, analytics and maintenance.
Bye
ROBO: Bye! take care..
```
