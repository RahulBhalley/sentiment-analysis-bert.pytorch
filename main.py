# -*- coding: utf-8 -*_

# NN library
import torch
import torch.nn as nn
import torch.optim as optim
# Bert model and its tokenizer
from transformers import BertTokenizer, BertModel
# Text data
from torchtext import data, datasets
# Numerical computation
import numpy as np
# standard library
import random
import time
# Configuration
from config import *

# Set random seed for reproducible experiments
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Get tokens for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
init_token_id = tokenizer.cls_token_id
eos_token_id  = tokenizer.sep_token_id
pad_token_id  = tokenizer.pad_token_id
unk_token_id  = tokenizer.unk_token_id

max_input_len = tokenizer.max_model_input_sizes['bert-base-uncased']

# Tokensize and crop sentence to 510 (for 1st and last token) instead of 512 (i.e. `max_input_len`)
def tokenize_and_crop(sentence):
  tokens = tokenizer.tokenize(sentence)
  tokens = tokens[:max_input_len - 2]
  return tokens

# Load the IMDB dataset and
# return (train_iter, valid_iter, test_iter) tuple
def load_data():
  text = data.Field(
    batch_first=True,
    use_vocab=False,
    tokenize=tokenize_and_crop,
    preprocessing=tokenizer.convert_tokens_to_ids,
    init_token=init_token_id,
    pad_token=pad_token_id,
    unk_token=unk_token_id
  )

  label = data.LabelField(dtype=torch.float)

  train_data, test_data  = datasets.IMDB.splits(text, label)
  train_data, valid_data = train_data.split(random_state=random.seed(SEED))

  print(f"training examples count: {len(train_data)}")
  print(f"test examples count: {len(test_data)}")
  print(f"validation examples count: {len(valid_data)}")

  label.build_vocab(train_data)

  train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
  )

  return train_iter, valid_iter, test_iter

# Get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 
# Build model
# 

bert_model = BertModel.from_pretrained('bert-base-uncased')

# Sentiment model containing pretrained BERT as backbone
# and two-GRU layers for analyzing the BERT hidden representation
# and a linear layer for classfification (the sigmoid is applied by the criterion during training).
import torch.nn as nn

class SentimentModel(nn.Module):
  def __init__(
    self,
    bert,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout
  ):
      
    super(SentimentModel, self).__init__()
    
    self.bert = bert
    embedding_dim = bert.config.to_dict()['hidden_size']
    self.rnn = nn.GRU(
      embedding_dim,
      hidden_dim,
      num_layers=n_layers,
      bidirectional=bidirectional,
      batch_first=True,
      dropout=0 if n_layers < 2 else dropout
    )
    self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, text):
    with torch.no_grad():
      embedded = self.bert(text)[0]
            
    _, hidden = self.rnn(embedded)
    
    if self.rnn.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
    else:
      hidden = self.dropout(hidden[-1,:,:])
    
    output = self.out(hidden)
    return output

model = SentimentModel(
  bert_model,
  HIDDEN_DIM,
  OUTPUT_DIM,
  N_LAYERS,
  BIDIRECTIONAL,
  DROPOUT
)
print(model)

# time taken for single epoch
def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

# computes accuracy
def binary_accuracy(preds, y):
  rounded_preds = torch.round(torch.sigmoid(preds))
  correct = (rounded_preds == y).float()
  acc = correct.sum() / len(correct)
  return acc

# training step
def train(model, iterator, optimizer, criterion):
  # stats
  epoch_loss = 0
  epoch_acc = 0
  # train mode
  model.train()
  
  for batch in iterator:
    # train step
    optimizer.zero_grad()
    predictions = model(batch.text).squeeze(1)
    loss = criterion(predictions, batch.label)
    acc = binary_accuracy(predictions, batch.label)
    loss.backward()
    optimizer.step()
    # stats
    epoch_loss += loss.item()
    epoch_acc += acc.item()
  
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

# evaluates the model on given iterator (either 
# train_iter, valid_iter, or test_iter)
def evaluate(model, iterator, criterion):
    
  epoch_loss = 0
  epoch_acc = 0
  # evaluation mode
  model.eval()
  
  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.text).squeeze(1)
      loss = criterion(predictions, batch.label)
      acc = binary_accuracy(predictions, batch.label)
      epoch_loss += loss.item()
      epoch_acc += acc.item()
      
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

# function to make sentiment prediction during inference
def predict_sentiment(model, tokenizer, sentence):
  model.eval()
  tokens = tokenizer.tokenize(sentence)
  tokens = tokens[:max_input_len - 2]
  indexed = [init_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_id]
  tensor = torch.LongTensor(indexed).to(device)
  tensor = tensor.unsqueeze(0)
  prediction = torch.sigmoid(model(tensor))
  return prediction.item()

if __name__ == "__main__":
  # Train BERT
  if TRAIN:
    # load data
    train_iter, valid_iter, test_iter = load_data()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss().to(device)
    model = model.to(device)

    best_val_loss = float('inf')

    for epoch in range(N_EPOCHS):
      # start time
      start_time = time.time()
      # train for an epoch
      train_loss, train_acc = train(model, train_iter, optimizer, criterion)
      valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
      # end time
      end_time = time.time()
      # stats
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      # save model if has validation loss
      # better than last one
      if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
      # stats
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    # Test
    model.load_state_dict(torch.load('model.pt'))
    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
  
  # Infer from BERT
  else:
    model.load_state_dict(torch.load('model.pt', map_location=device))
    sentiment = predict_sentiment(model, tokenizer, TEXT)
    print(sentiment)