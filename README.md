# Assignment: Sentiment Analysis with BERT Model

## Problem statement

## Solution

## How to run this code

Clone this repo:

```bash
git clone https://github.com/rahulbhalley/rahulbhalley.git
cd rahulbhalley
```

### Install the dependencies

All dependencies are listed in requirements.txt. Please install them with the following command.

```bash
sudo pip3 install -r requirements.txt
```

### Inference

To make sentiment predictions simply change the argument `TRAIN = False` and run the following code:

```bash
python3 main.py --text 'It feel good to be back on building NLP programs.'
```

This will output the sentiment of this sentence.