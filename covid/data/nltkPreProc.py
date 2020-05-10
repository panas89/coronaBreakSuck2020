import re, unicodedata
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

#------------------- Text Methods ------------------

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub(r'\[.*?\]', '', text)

def remove_between_parenteses(text):
    return re.sub(r'\(.*?\)', '', text)

def remove_bad_phrases(text, bad_phrases):
    for phrase in bad_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    return text

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def tokenize_text(text):
    return word_tokenize(text)


#------------------- Token Methods ---------------------

def remove_non_ascii(tokens):
    """Remove non-ASCII characters from list of tokenized words"""
    return [unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            for token in tokens]

def to_lowercase(tokens):
    """Convert all characters to lowercase from list of tokenized words"""
    return [token.lower() for token in tokens]

def remove_punctuation(tokens):
    """Remove punctuation from list of tokenized words"""
    new_tokens = []
    for token in tokens:
        new_token = re.sub(r'[^\w\s]', '', token)
        if new_token != '':
            new_tokens.append(new_token)
    return new_tokens

def remove_low_char(tokens, min_char):
    return [token for token in tokens if len(token) >= min_char]

def remove_digits(tokens):
    return [token for token in tokens if token.isalpha()]

def replace_numbers(tokens):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_tokens = []
    for token in tokens:
        if token.isdigit():
            try:
                new_token = p.number_to_words(token)
                new_tokens.append(new_token)
            except Exception as e:
                print(e)
        else:
            new_token.append(token)
    return new_token
    
def remove_stopwords(tokens, bad_tokens=None):
    """Remove stop words from list of tokenized words"""
    remove_tokens = stopwords.words('english')
    if bad_tokens:
        remove_tokens.extend(bad_tokens)
    return [token for token in tokens if token not in remove_tokens]

def stem_tokens(tokens):
    """Stem tokens in list of tokenized words"""
    stemmer = LancasterStemmer()
    return [stemmer.stem(token) for token in tokens]

def lemmatize_tokens(tokens, pos_tag='v'):
    """Lemmatize tokens in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, pos=pos_tag) for token in tokens]

#-------------------- Text PreProcessor ---------------------------


def preprocess_text(text, bad_phrases=None, bad_tokens=None, min_char=3, 
                    pos_tags=['v'], remove_dig=True, replace_num=False,
                    replace_contr=False):
    """
    Input: text (string)
    Output: list of tokens (strings)
    Parameters:
        - bad_phrases: list of regex - if not None, then remove all listed phrases from text
        - bad_tokens: list of strings - if not None, then remove all listed strings from tokens
        - min_char: integer - remove all token with characters strictly less than min_char
        - pos_tags: list of pos tags - lemmatize tokens with listed pos tags
        - remove_dig: Boolean - if True, remove all digits from tokens
        - replace_num: Boolean - if remove_digits=False and replace_numbers=True, convert digits to words
        - replace_contr: Boolean - if True, replace e.g. you're -> you are
    """

    # A. Text Methods
    #--------------------
    # 1. Remove urls
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Remove terms in brackets
    text = remove_between_square_brackets(text)
    text = remove_between_parenteses(text)

    # 3. Remove common phrases
    if bad_phrases:
        text = remove_bad_phrases(text, bad_phrases)

    # 4. Replace contractions s.t. you're -> you are
    if replace_contr:
        text = replace_contractions(text)

    # 5. Tokenize text
    tokens= tokenize_text(text)

    # B. Token Methods
    #--------------------

    # 1. Covert tokens to lowercase 
    tokens = to_lowercase(tokens)

    # 2. Remove non-ASCII strings
    tokens = remove_non_ascii(tokens)

    # 3. Remove Stopwords & bad tokens
    tokens = remove_stopwords(tokens, bad_tokens=bad_tokens)

    # 4. Remove punctuation
    tokens = remove_punctuation(tokens)

    # 5. Filter tokens of length < min_char
    tokens = remove_low_char(tokens, min_char)

    # 6. Remove or Convert digits to words
    if remove_dig:
        tokens = remove_digits(tokens)
    elif replace_num:
        tokens = replace_numbers(tokens)

    # 7. Lemmatize tokens
    for tag in pos_tags:
        tokens = lemmatize_tokens(tokens, pos_tag=tag)

    # 8. Repeat filter stop/bad tokens
    tokens = remove_stopwords(tokens, bad_tokens=bad_tokens)


    return tokens