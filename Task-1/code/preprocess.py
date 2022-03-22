
"""
### Preprocessing: 

This file contains functions we use for preprocessing the advertising data. The steps used are as follows:


*   Expand Contractions 
*   Remove special characters
*   Convert to lowecase 
*   Remove stop words 
*   Reduce term variations by stemming 
*   Remove ponctuation 
*   Numerically represent the text data using a proper tokenizer

We also have a function to split the data into training and evaluation sets
"""

import pandas as pd 
import string
import re 

# we use the nltk package for preprocessing step 
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# we use the sckitlearn package for splitting the datasets to training and evaluation sets same for feature extraction
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
contraction_dict = {"ain't": "are not","'s":" is","aren't": "are not"}

ps = PorterStemmer()

# Functions for removing contractions 
def expand_contractions(data,contractions_dict = contraction_dict):
  '''
    Expanding Contractions
    Arguments:
      data: textual dataset 
      contractions_dict : dictionanary containing the contractions and their replacements 
    Returns :
      clean_data : textual dataset where contractions are expanded
  '''
  # Regular expression for finding contractions
  contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
  def replace(match):
      return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, data)

# tokenize the data numerically 
# Each document will be represented using the bag of Words model or BoW
def tokenized_tfidf(data):
  '''
    Arguments :
      data : textual data 

    Returns : 
      tokenized textual data using the TF-IDF score
  '''
  count_vect = CountVectorizer()
  x_counts = count_vect.fit_transform(data)
  tfidf_transformer = TfidfTransformer()
  return tfidf_transformer.fit_transform(x_counts)

# preprocessing funtion 
def preprocess_data(data):
  '''
    Arguments :
      data : textual data 
    
    Returns : 
      clean_data: clean textual data to use for training various models
  '''
  # Lowercase 
  lowercase_data = data.str.strip().str.lower() 

  # Expand contractions 
  expand_contract_data = lowercase_data.apply(lambda x:expand_contractions(x))
  
  # Remove ponctuation 
  remove_ponct_data =  expand_contract_data.str.translate(str.maketrans('','',string.punctuation))

  # Remove stop words 
  remove_stp_wrd_data = remove_ponct_data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words("english"))]))

  return remove_stp_wrd_data

# training and evaluation data splitting 
# we use sckitlear's function

def eval_train_split(data,labels,val_size=0.2,test_size =0.2, random_state =42,validation = True):
  '''
    Arguments :

      data         : text data 
      labels       : the data labels 
      test_size    : defines what is the porportion of test to train data 
      random_state : Controls the shuffling applied to the data before applying the split, 
                     parameter for the scklearn train_test_split 
    Returns :
      x_train : the training data 
      y_train : labels for the training data 
      x_val :   the validation data 
      y_val :   labels for the validation data
      x_test  : the evaluation data 
      y_test  : labels for the evaluation data 

  '''  
  x_, x_test, y_, y_test = train_test_split(data, labels , test_size= test_size, random_state= random_state)

  if(validation):
    x_train,x_val,y_train,y_val = train_test_split(x_,y_, test_size=val_size, random_state= random_state)
    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))
  else:
     return ((x_, y_), (x_test, y_test))

def get_data(file, columns_to_drop, drop = False):
  '''
    Arguments :
      file : a string for the csv file path 
      columns_to_drop : list of columns to drop 
    
    Returns : 
       dataframe (texts and labels)
  '''

  df = pd.read_csv(file)
  if drop:
    df = df.drop(columns_to_drop, axis=1)

  return df