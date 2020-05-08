# Assignment: Sentiment Analysis with BERT Model

## Problem statement
The goal is to train a deep neural network to predict the sentiment of (hate) speech text. 

## Solution
This problem was solved by training a recent text classification model called BERT (Bidirectional Encoder Representations from Transformers). 

The pretrained model was loaded from Hugging Faces' `transformers`. Then a two layer bidirectional `nn.GRU` sub-networks was appended at BERT's out. When input tokenized ids of text are passed through BERT then it outputs a representation (also called embedding in this case). This representation is then passed through GRU layers. Following the time-step from GRU the last time-step's hidden state is extracted and then passed through a logistic classifier (using `nn.Linear` module). Following this we train the model using `nn.BCEWithLogitsLoss` loss.

We used IMDB dataset instead of using some specific hate speech dataset. Surprisingly the model still nicely classifies the hate speech.

### Future work
1. Training on an actual hate speech (text) dataset, instead of IMDB sentiment dataset, will give better accuracy for predicting sentiment of hate speech.
1. This BERT model can be trained longer to see if the accuracy improves further.
3. One can choose to train some other recent SOTA Transformer models such as Longformer, Electra, etc. for sentiment classification.

## Results

Train accuracy: 93.17%

Train loss: 0.180

Test accuracy: 92.24%

Test loss: 0.198

Validation accuracy: 91.26%

Validation loss: 0.219

## How to run this code

Clone this repo:

```bash
git clone https://github.com/rahulbhalley/rahulbhalley.git
cd rahulbhalley
```

### Install the dependencies

All dependencies are listed in `requirements.txt`. Please install them with the following command.

```bash
sudo pip3 install -r requirements.txt
```

### Usage

For both training and inferencing the BERT model you only need to run the following code.

```bash
python3 main.py
```

All you need to do is make changes to variables in `config.py` for controlling the behavior of this program.

Following is a list of settings and their descriptions:

- `TRAIN`: set `True` to train BERT model but for inference set it to `False`
- `TEXT`: a string whose sentiment is predicted when infering the model
- `BATCH_SIZE`: (default: 128) batch size of train examples to use during training.
- `N_EPOCHS`: (default: 05) total epochs to train the model for.
- `SEED`: (default: 12345) the seed value to make experiments reproducible.
- `BIDIRECTIONAL`: set `True` for bidirectional `nn.GRU` otherwise `False`.
- `HIDDEN_DIM`: (default: 256)hidden dimensions for `nn.GRU`.
- `OUTPUT_DIM`: (default: 1, don't change) since sentiment is binary output.
- `N_LAYERS`: (default: 2) number of layers of `nn.GRU` layer.
- `DROPOUT`: (default: 0.25) dropout rate.

For example, to make sentiment predictions (i.e. inference) simply set `TRAIN = False` in `config.py`. Also provide the text to `TEXT` variable in `config.py`. This will load the trained parameters `model.pt` and make prediction.

**Note**: The output closer to `1` means hate speech whereas in case of close to `0` means the content of text is good/happy.

For `TEXT = 'I like you!'`, the sentiment is closer to `0` (exactly `0.13519087433815002`). 

But when `TEXT = 'I hate you!'` then the sentiment is close to `1` (exactly `0.7887630462646484`).

In order to train the model just keep `TRAIN = True` in `config.py` and run `python3 main.py`.