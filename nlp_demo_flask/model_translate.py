import unicodedata
import re
import math
import psutil
import time
import datetime
from io import open
import random
from random import shuffle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.cuda

"""this line clears sys to allow for argparse to work as gradient clipper"""
#import sys; sys.argv=['']; del sys

"""This function converts a Unicode string to plain ASCII 
from https://stackoverflow.com/a/518232/2809427"""
def uniToAscii(sentence):
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
    )


"""Lowercase, trim, and remove non-letter characters (from pytorch)"""
def normalizeString(s):
    s = re.sub(r" ##AT##-##AT## ", r" ", s)
    s = uniToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s   

  
"""Denote patterns that sentences must start with to be kept in dataset. 
Can be changed if desired (from pytorch)"""



"""Filters each input-output pair, keeping sentences that are less than max_length 
if start_filter is true, also filters out sentences that don't start with eng_prefixes"""
def filterPair(p, max_length, start_filter):
    eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ")
    filtered = len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length 
    if start_filter:
        return filtered and p[1].startswith(eng_prefixes)
    else:
        return filtered

"""Filters all of the input-output language pairs in the dataset using filterPair 
for each pair (from pytorch)"""
def filterPairs(pairs, max_length, start_filter):
    return [pair for pair in pairs if filterPair(pair, max_length, start_filter)]

"""start of sentence tag"""
#SOS_token = 0

"""end of sentence tag"""
#EOS_token = 1

"""unknown word tag (this is used to handle words that are not in our Vocabulary)"""
#UNK_token = 2


"""Lang class, used to store the vocabulary of each language"""
class Lang:
    def __init__(self, language):
        self.language_name = language
        self.word_to_index = {"SOS":0, "EOS":1, "<UNK>":2}
        self.word_to_count = {}
        self.index_to_word = {0: "SOS", 1: "EOS", 2: "<UNK>"}
        self.vocab_size = 3
        self.cutoff_point = -1


    def countSentence(self, sentence):
        for word in sentence.split(' '):
            self.countWords(word)

    """counts the number of times each word appears in the dataset"""
    def countWords(self, word):
        if word not in self.word_to_count:
            self.word_to_count[word] = 1
        else:
            self.word_to_count[word] += 1

    """if the number of unique words in the dataset is larger than the
    specified max_vocab_size, creates a cutoff point that is used to
    leave infrequent words out of the vocabulary"""
    def createCutoff(self, max_vocab_size):
        word_freqs = list(self.word_to_count.values())
        word_freqs.sort(reverse=True)
        if len(word_freqs) > max_vocab_size:
            self.cutoff_point = word_freqs[max_vocab_size]

    """assigns each unique word in a sentence a unique index"""
    def addSentence(self, sentence):
        new_sentence = ''
        for word in sentence.split(' '):
            unk_word = self.addWord(word)
            if not new_sentence:
                new_sentence =unk_word
            else:
                new_sentence = new_sentence + ' ' + unk_word
        return new_sentence

    """assigns a word a unique index if not already in vocabulary
    and it appeaars often enough in the dataset
    (self.word_to_count is larger than self.cutoff_point)"""
    def addWord(self, word):
        if self.word_to_count[word] > self.cutoff_point:
            if word not in self.word_to_index:
                self.word_to_index[word] = self.vocab_size
                self.index_to_word[self.vocab_size] = word
                self.vocab_size += 1
            return word
        else:
            return self.index_to_word[2]


def prepareLangs(lang1, lang2, file_path, reverse=False):
   

    if len(file_path) == 2:
        lang1_lines = open(file_path[0], encoding='utf-8').\
            read().strip().split('\n')

        lang2_lines = open(file_path[1], encoding='utf-8').\
            read().strip().split('\n')

        if len(lang1_lines) != len(lang2_lines):
            quit()

        pairs = []

        for line in range(len(lang1_lines)):
            pairs.append([normalizeString(lang1_lines[line]),
                          normalizeString(lang2_lines[line])])            


    elif len(file_path) == 1:
        lines = open(file_path[0], encoding='utf-8').\
    	read().strip().split('\n')
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



"""completely prepares both input and output languages 
and returns cleaned and trimmed train and test pairs"""
#done
def prepareData(lang1, lang2, file_path, max_vocab_size=50000, 
                reverse=False, trim=0, start_filter=False, perc_train_set=0.9, 
                print_to=None):
    
    input_lang, output_lang, pairs = prepareLangs(lang1, lang2, 
                                                  file_path, reverse)
    
   
    
    if print_to:
        with open(print_to,'a') as f:
            f.write("Read %s sentence pairs \n" % len(pairs))
    
    if trim != 0:
        pairs = filterPairs(pairs, trim, start_filter)
        if print_to:
            with open(print_to,'a') as f:
                f.write("Read %s sentence pairs \n" % len(pairs))

    for pair in pairs:
        input_lang.countSentence(pair[0])
        output_lang.countSentence(pair[1])


    input_lang.createCutoff(max_vocab_size)
    output_lang.createCutoff(max_vocab_size)

    pairs = [(input_lang.addSentence(pair[0]),output_lang.addSentence(pair[1])) 
             for pair in pairs]

    shuffle(pairs)
    
    train_pairs = pairs[:math.ceil(perc_train_set*len(pairs))]
    test_pairs = pairs[math.ceil(perc_train_set*len(pairs)):]

    if print_to:
        with open(print_to,'a') as f:
            f.write("Train pairs: %s" % (len(train_pairs)))
            f.write("Test pairs: %s" % (len(test_pairs)))
            f.write("Counted Words -> Trimmed Vocabulary Sizes (w/ EOS and SOS tags):")
            f.write("%s, %s -> %s" % (input_lang.language_name, 
                                      len(input_lang.word_to_count),
                                      input_lang.vocab_size,))
            f.write("%s, %s -> %s \n" % (output_lang.language_name, len(output_lang.word_to_count), 
                            output_lang.vocab_size))
        
    return input_lang, output_lang, train_pairs, test_pairs


"""converts a sentence to one hot encoding vectors - pytorch allows us to just
use the number corresponding to the unique index for that word,
rather than a complete one hot encoding vector for each word"""
#done
def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        try:
            indexes.append(lang.word_to_index[word])
        except:
            indexes.append(lang.word_to_index["<UNK>"])
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    use_cuda = torch.cuda.is_available()
    EOS_token = 1
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1)
    if use_cuda:
        return result.cuda()
    else:
        return result
      
"""converts a pair of sentence (input and target) to a pair of tensors"""
def tensorsFromPair(input_lang, output_lang, pair):
    input_variable = tensorFromSentence(input_lang, pair[0])
    target_variable = tensorFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)
  

"""converts from tensor of one hot encoding vector indices to sentence"""
def sentenceFromTensor(lang, tensor):
    raw = tensor.data
    words = []
    for num in raw:
        words.append(lang.index_to_word[num.item()])
    return ' '.join(words)


"""seperates data into batches of size batch_size"""
#done
def batchify(data, input_lang, output_lang, batch_size, shuffle_data=True):
    if shuffle_data == True:
        shuffle(data)
    number_of_batches = len(data) // batch_size
    batches = list(range(number_of_batches))
    longest_elements = list(range(number_of_batches))
    
    for batch_number in range(number_of_batches):
        longest_input = 0
        longest_target = 0
        input_variables = list(range(batch_size))
        target_variables = list(range(batch_size))
        index = 0      
        for pair in range((batch_number*batch_size),((batch_number+1)*batch_size)):
            input_variables[index], target_variables[index] = tensorsFromPair(input_lang, output_lang, data[pair])
            if len(input_variables[index]) >= longest_input:
                longest_input = len(input_variables[index])
            if len(target_variables[index]) >= longest_target:
                longest_target = len(target_variables[index])
            index += 1
        batches[batch_number] = (input_variables, target_variables)
        longest_elements[batch_number] = (longest_input, longest_target)
    return batches , longest_elements, number_of_batches


"""pads batches to allow for sentences of variable lengths to be computed in parallel"""
def pad_batch(batch):
    EOS_token = 1
    padded_inputs = torch.nn.utils.rnn.pad_sequence(batch[0],padding_value=EOS_token)
    padded_targets = torch.nn.utils.rnn.pad_sequence(batch[1],padding_value=EOS_token)
    return (padded_inputs, padded_targets)




class EncoderRNN(nn.Module):
	def __init__(self,input_size,hidden_size,layers=1,dropout=0.1,
               bidirectional=True):
		super(EncoderRNN, self).__init__()

		if bidirectional:
			self.directions = 2
		else:
			self.directions = 1
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = layers
		self.dropout = dropout
		self.embedder = nn.Embedding(input_size,hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,
                        num_layers=layers,dropout=dropout,
                        bidirectional=bidirectional,batch_first=False)
		self.fc = nn.Linear(hidden_size*self.directions, hidden_size)

	def forward(self, input_data, h_hidden, c_hidden):
		embedded_data = self.embedder(input_data)
		embedded_data = self.dropout(embedded_data)
		hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))

		return hiddens, outputs

	"""creates initial hidden states for encoder corresponding to batch size"""
	def create_init_hiddens(self, batch_size):
		h_hidden = Variable(torch.zeros(self.num_layers*self.directions, 
                                    batch_size, self.hidden_size))
		c_hidden = Variable(torch.zeros(self.num_layers*self.directions, 
                                    batch_size, self.hidden_size))
		if torch.cuda.is_available():
			return h_hidden.cuda(), c_hidden.cuda()
		else:
			return h_hidden, c_hidden
        

#done
class DecoderAttn(nn.Module):
	def __init__(self, hidden_size, output_size, layers=1, dropout=0.1, bidirectional=True):
		super(DecoderAttn, self).__init__()

		if bidirectional:
			self.directions = 2
		else:
			self.directions = 1
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.num_layers = layers
		self.dropout = dropout
		self.embedder = nn.Embedding(output_size,hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.score_learner = nn.Linear(hidden_size*self.directions, 
                                   hidden_size*self.directions)
		self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,
                        num_layers=layers,dropout=dropout,
                        bidirectional=bidirectional,batch_first=False)
		self.context_combiner = nn.Linear((hidden_size*self.directions)
                                      +(hidden_size*self.directions), hidden_size)
		self.tanh = nn.Tanh()
		self.output = nn.Linear(hidden_size, output_size)
		self.soft = nn.Softmax(dim=1)
		self.log_soft = nn.LogSoftmax(dim=1)


	def forward(self, input_data, h_hidden, c_hidden, encoder_hiddens):

		embedded_data = self.embedder(input_data)
		embedded_data = self.dropout(embedded_data)	
		batch_size = embedded_data.shape[1]
		hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))	
		top_hidden = outputs[0].view(self.num_layers,self.directions,
                                 hiddens.shape[1],
                                 self.hidden_size)[self.num_layers-1]
		top_hidden = top_hidden.permute(1,2,0).contiguous().view(batch_size,-1, 1)

		prep_scores = self.score_learner(encoder_hiddens.permute(1,0,2))
		scores = torch.bmm(prep_scores, top_hidden)
		attn_scores = self.soft(scores)
		con_mat = torch.bmm(encoder_hiddens.permute(1,2,0),attn_scores)
		h_tilde = self.tanh(self.context_combiner(torch.cat((con_mat,
                                                         top_hidden),dim=1)
                                              .view(batch_size,-1)))
		pred = self.output(h_tilde)
		pred = self.log_soft(pred)

		
		return pred, outputs
    

'''Returns the predicted translation of a given input sentence. Predicted
translation is trimmed to length of cutoff_length argument'''

def evaluate(encoder, decoder, sentence, input_lang, output_lang, cutoff_length):
	with torch.no_grad():
		input_variable = tensorFromSentence(input_lang, sentence)
		input_variable = input_variable.view(-1,1)
		enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(1)

		enc_hiddens, enc_outputs = encoder(input_variable, enc_h_hidden, enc_c_hidden)

		decoder_input = Variable(torch.LongTensor(1,1).fill_(output_lang.word_to_index.get("SOS")))
		dec_h_hidden = enc_outputs[0]
		dec_c_hidden = enc_outputs[1]

		decoded_words = []

		for di in range(cutoff_length):
			pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

			topv, topi = pred.topk(1,dim=1)
			ni = topi.item()
			if ni == output_lang.word_to_index.get("EOS"):
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(output_lang.index_to_word[ni])

			decoder_input =  Variable(torch.LongTensor(1,1).fill_(ni))
			dec_h_hidden = dec_outputs[0]
			dec_c_hidden = dec_outputs[1]

		output_sentence = ' '.join(decoded_words)
    
		return output_sentence


'''Evaluates prediction translations for a specified number (n) of sentences
chosen randomly from a list of passed sentence pairs. Returns three sentences
in the format:
                  > input sentence
                  = correct translation
                  < predicted translation'''
def evaluate_randomly(encoder, decoder, pairs, n=2, trim=100):
	for i in range(n):
		pair = random.choice(pairs)
		output_sentence = evaluate(encoder, decoder, pair[0],cutoff_length=trim)
		return output_sentence
		 
            


'''Used to plot the progress of training. Plots the loss value vs. time'''
def showPlot(times, losses, fig_name):
    x_axis_label = 'Minutes'
    colors = ('red','blue')
    if max(times) >= 120:
    	times = [mins/60 for mins in times]
    	x_axis_label = 'Hours'
    i = 0
    for key, losses in losses.items():
    	if len(losses) > 0:
    		plt.plot(times, losses, label=key, color=colors[i])
    		i += 1
    plt.legend(loc='upper left')
    plt.xlabel(x_axis_label)
    plt.ylabel('Loss')
    plt.title('Training Results')
    plt.savefig(fig_name+'.png')
    plt.close('all')
    
'''prints the current memory consumption'''
#main_func
def function_translate(sentence):
  use_cuda = torch.cuda.is_available()
  #todo
  input_lang_name= 'sp'
  output_lang_name = 'en'
  dataset = 'orig'
  raw_data_file_path = ('spa.txt',)

  reverse=True
  trim = 10
  max_vocab_size= 20000
  start_filter = False
  perc_train_set = 0.8
  """for plotting of the loss"""
  plt.switch_backend('agg')

  output_file_name = ""

  print_to = None
  bidirectional = True
  if bidirectional:
    	directions = 2
  else:
    	directions = 1
    
    
  layers = 2
    
   
  hidden_size = 440
    
  dropout = 0.2
    
   

    
    

  input_lang, output_lang, train_pairs, test_pairs = prepareData(
      input_lang_name, output_lang_name, raw_data_file_path, 
      max_vocab_size=max_vocab_size, reverse=reverse, trim=trim, 
      start_filter=start_filter, perc_train_set=perc_train_set, print_to=print_to)

  """create the Encoder"""
  encoder = EncoderRNN(input_lang.vocab_size, hidden_size, layers=layers, 
                      dropout=dropout, bidirectional=bidirectional)

  """create the Decoder"""
  decoder = DecoderAttn(hidden_size, output_lang.vocab_size, layers=layers, 
                        dropout=dropout, bidirectional=bidirectional)

  encoder.load_state_dict(torch.load('enc_weights.pt', map_location=torch.device('cpu')))
  decoder.load_state_dict(torch.load('dec_weights.pt', map_location=torch.device('cpu')))

  if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
  outside_sent = normalizeString(sentence)
  str=""
  str = evaluate(encoder, decoder, outside_sent, input_lang, output_lang, cutoff_length=10)
  return str
