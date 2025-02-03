#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:06:05 2025

@author: qb
"""

import torch
from torch import nn
import random

class Seq2SeqPackedAttention(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device, additive):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device  = device
        self.additive = additive
    def create_mask(self, src):
        #src: [src len, batch_size]
        mask = torch.isin(src, torch.tensor(self.src_pad_idx))
        mask = mask.permute(1, 0)  #permute so that it's the same shape as attention
        #mask: [batch_size, src len] #(0, 0, 0, 0, 0, 1, 1)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0):
        #src: [src len, batch_size]
        #trg: [trg len, batch_size]
        
        #initialize something
        batch_size = src.shape[1]
        trg_len    = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs    = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)
        
        #send our src text into encoder
        encoder_outputs, hidden = self.encoder(src)
        #encoder_outputs refer to all hidden states (last layer)
        #hidden refer to the last hidden state (of each layer, of each direction)
        
        input_ = trg[0, :]
        
        mask   = self.create_mask(src) #(0, 0, 0, 0, 0, 1, 1)
        
        #for each of the input of the trg text
        for t in range(1, trg_len):
            #send them to the decoder
            output, hidden, attention = self.decoder(input_, hidden, encoder_outputs, mask)
            #output: [batch_size, output_dim] ==> predictions
            #hidden: [batch_size, hid_dim]
            #attention: [batch_size, src len]
            
            #append the output to a list
            outputs[t] = output
            attentions[t] = attention
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1          = output.argmax(1)  #autoregressive
            
            input_ = trg[t] if teacher_force else top1
            
        return outputs, attentions




class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Embedding the input sequence
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]

        # Pass the embedded sequence through the GRU
        outputs, hidden = self.rnn(embedded)  # [src_len, batch_size, hid_dim * 2]

        # We concatenate the last hidden states from both forward and backward directions
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))  # [batch_size, hid_dim]

        return outputs, hidden


    
    
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.v = nn.Linear(hid_dim, 1, bias = False)
        self.W = nn.Linear(hid_dim, hid_dim) #for decoder input_
        self.U = nn.Linear(hid_dim * 2, hid_dim)  #for encoder_outputs

    
    def forward(self, hidden, encoder_outputs, mask):
        #hidden = [batch_size, hid_dim] ==> first hidden is basically the last hidden of the encoder
        #encoder_outputs = [src len, batch_size, hid_dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len    = encoder_outputs.shape[0]
        
        #repeat the hidden src len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch_size, src_len, hid_dim]
        
        #permute the encoder_outputs just so that you can perform multiplication / addition
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, src_len, hid_dim * 2]
        
        #add
        energy = self.v(torch.tanh(self.W(hidden) + self.U(encoder_outputs))).squeeze(2)
        #(batch_size, src len, 1) ==> (batch_size, src len)
        
        #mask
        energy = energy.masked_fill(mask, -1e10)
        
        return torch.nn.functional.softmax(energy, dim = 1)
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention  = attention
        self.embedding  = nn.Embedding(output_dim, emb_dim)
        self.rnn        = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc         = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input: [batch_size]
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]
        # mask: [batch_size, src_len] (mask for padded tokens)
        
        # Embed our input
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]
        
        # Apply attention mechanism, considering padding
        a = self.attention(hidden, encoder_outputs, mask)  # Attention weights: [batch_size, src_len]
        att = a.unsqueeze(1)  # [batch_size, 1, src_len]
        
        # Permute encoder outputs to [batch_size, src_len, hid_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim * 2]
        
        # Calculate weighted sum of encoder outputs based on attention weights
        weighted = torch.bmm(att, encoder_outputs)  # [batch_size, 1, hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)  # [1, batch_size, hid_dim * 2]
        
        # Concatenate embedded input and weighted encoder outputs
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, emb_dim + hid_dim * 2]
        
        # Send input to the GRU
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))  # [1, batch_size, hid_dim]
        
        # Squeeze the outputs from the GRU and weighted sum
        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]
        output = output.squeeze(0)  # [batch_size, hid_dim]
        weighted = weighted.squeeze(0)  # [batch_size, hid_dim * 2]
        
        # Apply the fully connected layer to predict the next token
        prediction = self.fc(torch.cat((embedded, output, weighted), dim=1))  # [batch_size, output_dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)
