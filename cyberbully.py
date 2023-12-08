import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


import re
import pickle
import spacy
import nltk
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import preprocess_text as pt
import language_tool_python
# from pycontractions.contractions import Contractions
import contractions

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import Trainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image

import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import docx2txt
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber


def get_term_list(path):
    '''
    Function to import term list file
    '''
    word_list = []
    with open(path,"r") as f:
        for line in f:
            word = line.replace("\n","").strip()
            word_list.append(word)
    return word_list

def get_vocab(corpus):
    '''
    Function returns unique words in document corpus
    '''
    # vocab set
    unique_words = set()

    # looping through each document in corpus
    for document in corpus:
        for word in document.split(" "):
            if len(word) > 2:
                unique_words.add(word)

    return unique_words

def create_profane_mapping(profane_words,vocabulary):

    # mapping dictionary
    mapping_dict = dict()

    # looping through each profane word
    for profane in profane_words:
        mapped_words = set()

        # looping through each word in vocab
        for word in vocabulary:
            # mapping only if ratio > 80
            try:
                if fuzz.ratio(profane,word) > 90:
                    mapped_words.add(word)
            except:
                pass

        # list of all vocab words for given profane word
        mapping_dict[profane] = mapped_words

    return mapping_dict

def replace_words(corpus,mapping_dict):
    '''
    Function replaces obfuscated profane words using a mapping dictionary
    '''

    processed_corpus = []

    # iterating over each document in the corpus
    for document in corpus:

        # splitting sentence to word
        comment = document.split()

        # iterating over mapping_dict
        for mapped_word,v in mapping_dict.items():

            # comparing target word to each comment word 
            for target_word in v:

                # each word in comment
                for i,word in enumerate(comment):
                    if word == target_word:
                        comment[i] = mapped_word

        # joining comment words
        document = " ".join(comment)
        document = document.strip()

        processed_corpus.append(document)

    return processed_corpus


example_text = "I'd not hate you"
example_data = {
    "text" : [example_text]
}
df = pd.DataFrame(example_data)

def text_preprocessing_pipeline(df=df,
                                remove_url=False,
                                remove_email=False,
                                remove_user_mention=False,
                                remove_html=False,
                                remove_space_single_char=False,
                                normalize_elongated_char=False,
                                normalize_emoji=False,
                                normalize_emoticon=False,
                                normalize_accented=False,
                                lower_case=False,
                                normalize_slang=False,
                                normalize_badterm=False,
                                spelling_check=False,
                                normalize_contraction=False,
                                term_list=False,
                                remove_numeric=False,
                                remove_stopword=False,
                                keep_pronoun=False,
                                remove_punctuation=False,
                                lemmatise=False
                               ):

    if remove_url:
        print('Text Preprocessing: Remove URL')
        df['text_check'] = df['text'].apply(lambda x: pt.remove_urls(x))

    if remove_email:
        print('Text Preprocessing: Remove email')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_emails(x))

    if remove_user_mention:
        print('Text Preprocessing: Remove user mention')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_mention(x))

    if remove_html:
        print('Text Preprocessing: Remove html element')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_html_tags(x))

    if remove_space_single_char:
        print('Text Preprocessing: Remove single spcae between single characters e.g F U C K')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_space_single_chars(x))

    if normalize_elongated_char:
        print('Text Preprocessing: Reduction of elongated characters')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_elongated_chars(x))

    if normalize_emoji:
        print('Text Preprocessing: Normalize and count emoji')
        df['emoji_counts'] = df['text_check'].apply(lambda x: pt.get_emoji_counts(x))
        df['text_check'] = df['text_check'].apply(lambda x: pt.convert_emojis(x))

    if normalize_emoticon:
        print('Text Preprocessing: Normalize and count emoticon')
        df['emoticon_counts'] = df['text_check'].apply(lambda x: pt.get_emoticon_counts(x))
        df['text_check'] = df['text_check'].apply(lambda x: pt.convert_emoticons(x))

    if normalize_accented:
        print('Text Preprocessing: Normalize accented character')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_accented_chars(x))

    if lower_case:
        print('Text Preprocessing: Convert to lower case')
        df['text_check'] = df['text_check'].apply(lambda x: str(x).lower())

    if normalize_slang:
        print('Text Preprocessing: Normalize slang')
        df['text_check'] = df['text_check'].apply(lambda x: pt.slang_resolution(x))


    if normalize_contraction:
        print('Text Preprocessing: Contraction to Expansion')
        # df['text_check'] = df['text_check'].apply(lambda x: ''.join(list(cont.expand_texts([x], precise=True))))
        df['text_check'] = df['text_check'].apply(lambda x: contractions.fix(x,slang=False))

    if remove_numeric: 
        print('Text Preprocessing: Remove numeric')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_numeric(x))

    if remove_punctuation:
        print('Text Preprocessing: Remove punctuations')
        df['text_check'] = df['text_check'].apply(lambda x: pt.remove_special_chars(x))

    if remove_stopword:
        print('Text Preprocessing: Remove stopword')
        if keep_pronoun:
            print('Text Preprocessing: and, keep Pronoun')
        df["text_check"] = df["text_check"].apply(lambda x: pt.remove_stopwords(x,keep_pronoun=keep_pronoun))

    # Remove multiple spaces
    print('Text Preprocessing: Remove multiple spaces')
    df['text_check'] = df['text_check'].apply(lambda x: ' '.join(x.split()))

    if lemmatise:
        print('Text Preprocessing: Lemmatization')
        df["text_check"] = df["text_check"].apply(lambda x: pt.make_base(x))

    # Make sure lower case for all again
    df['text_check'] = df['text_check'].apply(lambda x: str(x).lower())

    # Remove empty text after cleaning
    print('Last Step: Remove empty text after preprocessing. Done')
    df = df[~df['text_check'].isna()]
    df = df[df['text_check'] != '']
    df = df.reset_index(drop=True)

    return df['text_check'].tolist()



@st.cache_resource()

def get_cb_model():

    # Define pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('teoh0821/cb_detection', num_labels=2)

    return tokenizer, model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

selected = option_menu(
    menu_title=None,
    options = ["Home", "Dataset", "Documents","Preprocess","Application"],
    icons = ["document","document","document","document","document"],
    default_index=0,
    orientation="horizontal",
    )

if selected =="Home":
		st.subheader("Welcome to Cyberbullying detection system")
		st.text("Features of this system:")
		st.info("1. Data uploading")
		st.info("2. Text-preprocessing")
		st.info("3. Cyberbully message detection")
  

if selected == "Dataset":
		st.subheader("Dataset")
		data_file = st.file_uploader("Upload CSV",type=['csv'])
		if st.button("Process", key=1):
			if data_file is not None:
				file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
				st.write(file_details)

				df = pd.read_csv(data_file)
				st.dataframe(df)
    
if selected =="Documents":
        st.subheader("Documents")
        docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
        if st.button("Process", key=2):
                if docx_file is not None:
                    file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
                    st.write(file_details)
                    if docx_file.type == "text/plain":
                        raw_text = str(docx_file.read(),"utf-8") 
                        st.write(raw_text)
                    elif docx_file.type == "application/pdf":
                        try:
                            with pdfplumber.open(docx_file) as pdf:
                                page = pdf.pages[0]
                                st.write(page.extract_text())
                        except:
                            st.write("None")
					    
					
                    elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text = docx2txt.process(docx_file) 
                        st.write(raw_text)

if selected =="Preprocess":
    st.subheader("Text preprocessing")
    input_text = st.text_area('Enter Text to preprocess')
    button = st.button("Analyze")
    if input_text and button:

        # Read data 
        input_data = {
                        "text" : [input_text] 
                    }

        bully_data = pd.DataFrame(input_data)

        
        cleaned_input_text = text_preprocessing_pipeline(
                                                df=bully_data,
                                                remove_url=True,
                                                remove_email=True,
                                                remove_user_mention=True,
                                                remove_html=False,
                                                remove_space_single_char=True,
                                                normalize_elongated_char=True,
                                                normalize_accented=True,
                                                lower_case=True,
                                                normalize_slang=True,
                                                normalize_contraction=True,
                                                remove_numeric=True,
                                                remove_stopword=False,
                                                keep_pronoun=True,  
                                                remove_punctuation=True,
                                                lemmatise=True)

        st.write(cleaned_input_text)
    
    
  

if selected == "Application":   
    st.subheader("Cyberbully message detection")
    
    with st.spinner("Setting up.."):
        tokenizer, model = get_cb_model()

    input_text = st.text_area('Enter Text to Analyze')
    button = st.button("Analyze")

    if input_text and button:

        input_data = {
                        "text" : [input_text] 
                    }

        bully_data = pd.DataFrame(input_data)

        with st.spinner("Hold on.. Preprocessing the input text.."):
            cleaned_input_text = text_preprocessing_pipeline(
                                                df=bully_data,
                                                remove_url=True,
                                                remove_email=True,
                                                remove_user_mention=True,
                                                remove_html=False,
                                                remove_space_single_char=True,
                                                normalize_elongated_char=True,
                                                normalize_accented=True,
                                                lower_case=True,
                                                normalize_slang=True,
                                                normalize_contraction=True,
                                                remove_numeric=True,
                                                remove_stopword=True,
                                                keep_pronoun=True,  
                                                remove_punctuation=True,
                                                lemmatise=True)

        with st.spinner("Almost there.. Analyzing your input text.."):
            input_text_tokenized = tokenizer(cleaned_input_text, padding=True, truncation=True, max_length=512)

            # Create torch dataset
            input_text_dataset = Dataset(input_text_tokenized)

            # Define test trainer
            pred_trainer = Trainer(model)

            # Make prediction
            raw_pred, _, _ = pred_trainer.predict(input_text_dataset)

            # Preprocess raw predictions
            text_pred = np.where(np.argmax(raw_pred, axis=1)==1,"Cyberbullying Post","Non-cyberbullying Post")

            if text_pred.tolist()[0] == "Non-cyberbullying Post":
                st.write("This is a non-cyberbullying message.")
            elif text_pred.tolist()[0] == "Cyberbullying Post":
                st.write("This may be a cyberbullying message!")
        
