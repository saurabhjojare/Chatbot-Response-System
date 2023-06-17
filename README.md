### [Presentation](https://docs.google.com/presentation/d/1LHFyjx8DyFmtC3ej8HHw4idewAuG6EcOgZIZ4suzJYw/edit?usp=sharing)
## Chatbot Response System in Python (using NLTK) 

The Intelligent Python Chatbot project aims to enhance the user experience by providing a conversational and intelligent interface capable of understanding and responding to user queries effectively. This project presents an opportunity to explore and implement NLP techniques and machine learning algorithms in the field of chatbot development.

## NLP
NLP is a way for computers to analyze, understand, and derive meaning from human language in a smart and useful way. By utilizing NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation.

## NLTK
NLTK (Natural Language Toolkit) is a powerful Python library widely used for natural language processing (NLP) tasks. It provides various tools and resources for tasks such as tokenization, stemming, lemmatization, part-of-speech tagging, syntactic parsing, and more. NLTK is open-source and offers a vast collection of text corpora, lexical resources, and pre-trained models for research and development in NLP.

## TF-IDF Approach
A problem with the Bag of Words approach is that highly frequent words start to dominate in the document (e.g. larger score), but may not contain as much “informational content”. Also, it will give more weight to longer documents than shorter documents.

One approach is to rescale the frequency of words by how often they appear in all documents so that the scores for frequent words like “the” that are also frequent across all documents are penalized. This approach to scoring is called Term Frequency-Inverse Document Frequency, or TF-IDF for short, where:

### Term Frequency: is a scoring of the frequency of the word in the current document.

&nbsp; &nbsp; TF = (Number of times term t appears in a document)/(Number of terms in the document)
 
### Inverse Document Frequency: is a scoring of how rare the word is across documents.

&nbsp; &nbsp; IDF = 1+log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
 
## Cosine Similarity

Tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus

&nbsp; &nbsp; Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||

where d1,d2 are two non zero vectors.

To generate a response from our bot for input questions, the concept of document similarity will be used. We define a function response which searches the user’s utterance for one or more known keywords and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, it returns a response:” I am sorry! I don’t understand you”

## History 

History of chatbots dates back to 1966 when a computer program called ELIZA was invented by Weizenbaum. It imitated the language of a psychotherapist from only 200 lines of code. You can still converse with it here: [Eliza](http://psych.fullerton.edu/mbirnbaum/psych101/Eliza.htm?utm_source=ubisend.com&utm_medium=blog-link&utm_campaign=ubisend). 
