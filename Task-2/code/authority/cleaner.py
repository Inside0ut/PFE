
from re import sub 
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import textblob as tb
from bs4 import BeautifulSoup

import nltk
nltk.download('stopwords')

#Importing wordnet 
nltk.download('wordnet')
nltk.download('omw-1.4')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

#Tokenization of text
tokenizer=ToktokTokenizer()


#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text= sub(pattern,'',text)
    return text

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

# Lemmatization of text
def simple_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text



# removing adjevtives 
def get_adjectives(text):
    blob = tb.TextBlob(text)
    result = [ word for (word,tag) in blob.tags if tag == "JJ"]
    return result 

def remove_adjectives(text):
    list_adjectives = get_adjectives(text) 
    for a in list_adjectives :
      text = text.replace(a, '')
    return text

