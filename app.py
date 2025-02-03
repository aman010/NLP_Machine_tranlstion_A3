#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:22:33 2025

@author: qb
"""

import streamlit as st
import torch  
import numpy as np
import nltk
import numpy as np
from torch import nn
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from additiveAttention import Seq2SeqPackedAttention, Encoder, Decoder, Attention
from nltk.corpus import stopwords 
import re   
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import torch.nn.functional as F
import random

# Title of the app

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

import torch
import torch.nn.functional as F

#ecoder object 

def greedy_decode_without_sos_eos(decoder, encoder_outputs, encoder_hidden, input_target_sequence,vocab, max_len=50, threshold=0.1):

    input_token = torch.tensor([0]).unsqueeze(0)  # Initial dummy token (or first token of source sentence)
    hidden = encoder_hidden
    
    decoded_sentence = [] # To store the decoded sentence
    vocab_size = len(vocab.tolist()['tar'])

    for i in range(vocab_size):
        input_token = input_target_sequence[i]
        
        mask = ~torch.isin(torch.tensor(input_token), torch.tensor(0)).bool()
        output, hidden, attention = decoder(torch.tensor(input_token), hidden, encoder_outputs, mask)
        print(output)
        
        probabilities = F.softmax(output, dim=-1)  # Shape: [batch_size, output_vocab_size]
        
        predicted_token = torch.argmax(probabilities, dim=-1).item()  # Get the token with max probability
        
        decoded_sentence.append(predicted_token)
        
        if probabilities[0, predicted_token] < threshold:
            break
        
    # Update the input_token for the next iteration
    input_token = torch.tensor([predicted_token]) 
        
    input_token = torch.tensor([predicted_token]).unsqueeze(0)  # Shape: [1, 1]
    
    return decoded_sentence
    
    
def pad_sequence(sequence, target_length, padding_value=0):
    # Convert to tensor if the sequence is a list
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)

    # If sequence is already of the target length, no padding is needed
    if sequence.size(0) >= target_length:
        return sequence[:target_length]
    
    # Calculate how much padding is required
    padding_size = target_length - sequence.size(0)
    
    # Create a padding tensor of the required size filled with the padding value
    padding_tensor = torch.full((padding_size,), padding_value, dtype=sequence.dtype)
    
    # Concatenate the sequence with the padding
    padded_sequence = torch.cat((sequence, padding_tensor), dim=0)
    
    return padded_sequence


st.title("English to Hindi Translation App")

# Subtitle
st.markdown("""
Simply input the text in English and click "Translate" to get the translation in Hindi The performance might be good still underwork.
""")

# Load your model (assuming PyTorch model here for illustration)
# If you're using another library like TensorFlow, replace this accordingly.
# Example: model = YourModel.load('path_to_your_model')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.load('path_to_your_model', map_location=device)
model_additive = torch.load('Model_corpa/umodel')
model_additive.eval()


input_src =np.load('Model_corpa/input_src.npy')
input_tar = np.load('Model_corpa/input_tar.npy')
vocab =np.load('Model_corpa/vocab_attention.npy', allow_pickle=True)

encoder=Encoder(len(vocab.tolist()['src']), 300, 540 ,0.3)
attn=Attention(540)
decoder = Decoder(len(vocab.tolist()['tar']),300, 540, attention=attn, dropout=0.3)

seq=Seq2SeqPackedAttention(encoder, decoder, 0, 'cpu', True)
seq.load_state_dict(model_additive.state_dict())


# Input text box (for the user to input English text)
source_text = st.text_area("Enter text in English to translate", height=150)

# Translation function (uses your model to perform inference)

def clean_and_preprocess_text(text):
    text = str(text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_w = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token not in stop_w]
    return tokens

   

def translate_text(input_text, selected_option):
    
    
    if not selected_option:
        input_text = clean_and_preprocess_text(input_text)
        print(input_text)
        src_vocab = vocab.tolist()['src']
        #we doing this way because to check if position can effect the prediction so we preserve the prediction
        r = []
        for text in input_text:
            if src_vocab[text] is None:
                r.append(0)
            r.append(src_vocab[text])
        print(r)
            
        s = [src_vocab[text] for text in input_text]
        
        if len(s) == 0:
            st.write("Vocab is Limited !")
            
        else:
            sr = torch.tensor(s)

            # if sr.shape[0] == 7:
        #     encoder_state, hidden = model_additive.encoder(torch.tensor(s))
        # else:
        #     psre = pad_sequence(sr.tolist(), 7)
        #     print(psre)
        #     encoder_state, hidden = model_additive.encoder(torch.tensor(psre).unsqueeze(0))

        target_vocab_size = len(vocab.tolist()['tar'])
        
        # decoded_sentence = greedy_decode_without_sos_eos(model_additive.decoder, encoder_state, hidden, max_len= target_vocab_size, 
        #                                                  threshold=0.5, 
        #                                                  input_target_sequence= input_tar,vocab = vocab )
        # print(decoded_sentence)
        
        seq.eval()
        rand = random.randint(0, input_tar.shape[0])
        psre = pad_sequence(sr.tolist(), 8)
        print('psr', psre)
        index_to_token = {i: token for i, token in enumerate(vocab.tolist()['tar'])}
        
        if not selection_options:
        
            with torch.no_grad():
                # opt = seq(psre.unsqueeze(0), 7, torch.tensor(input_tar[rand]).unsqueeze(0),0)
                encoder_state, hidden=encoder(psre.unsqueeze(0))
                mask =seq.create_mask(psre.unsqueeze(0))
                predictions, hidden, attention =decoder(torch.tensor(input_tar[rand]),hidden, encoder_state, mask = mask)
                output_probs = torch.softmax(predictions, dim=-1) 
                predicted_tokens = output_probs.argmax(dim=-1)
                predicted_sentence = [index_to_token[token.item()] for token in predicted_tokens]  
                st.write(predicted_sentence)
       
    else:
        seq.eval()
        index_to_token = {i: token for i, token in enumerate(vocab.tolist()['tar'])}
        psr = input_src[selection_options]
        encoder_state, hidden= encoder(torch.tensor(psr).unsqueeze(0))
        mask =seq.create_mask(torch.tensor(psr).unsqueeze(0))
        predictions, hidden, attention =decoder(torch.tensor(input_tar[selection_options]),hidden, encoder_state, mask = mask)
        output_probs = torch.softmax(predictions, dim=-1) 
        predicted_tokens = output_probs.argmax(dim=-1)
        predicted_sentence = [index_to_token[token.item()] for token in predicted_tokens]
        
        st.write(predicted_sentence)
        # For batch_size=1, get the first sentence
            
            # input_sentebce=[index_to_token_src[token] for token in s.squeeze(0).tolist()]
            
            
        print('output',predictions)
    #pass this s to the model to get the prediction
        
    return predicted_sentence
        
    #numerize the text from the vocab
    
    # Example of model inference; adjust according to your model's API
    # If using a custom pre-processing step, do it here (e.g., tokenization)
    
    # Example for PyTorch-based model
    # input_tensor = preprocess(input_text)  # Your preprocessing steps here
    # output = model(input_tensor)
    # translated_text = postprocess(output)  # Your postprocessing (e.g., detokenization)
    
    # Assuming your model output is just a string after translation:
    translated_text = "Translation output goes here"  # Replace with actual inference output
    return translated_text

# Button to trigger translation

options = list(range(0, input_src.shape[0]))
selection_options = st.selectbox("Select the source index", options)
index_to_token_src = {i: token for i, token in enumerate(vocab.tolist()['src'])}

if selection_options:
    src=input_src[selection_options]
    sent =[index_to_token_src[i] for i in src]
    st.write(sent)

if st.button("Translate", key="button_key"):
    if source_text:
        
        translation = translate_text(str(source_text), selection_options)
        st.success(f"Translated Text: {translation}")
    elif selection_options:
        translation = translate_text(str(source_text), selection_options)
        

        

