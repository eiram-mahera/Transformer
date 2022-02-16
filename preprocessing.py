import re
import pandas as pd
import tensorflow as tf


MAX_SENTENCE_LENGTH = 15
BATCH_SIZE = 64
src_lang_file = "data/news-commentary-v8.de-en.de"
trg_lang_file = "data/news-commentary-v8.de-en.en"
TRAIN_SIZE = 10000
VAL_SIZE = 1000
TEST_SIZE = 1000
MAX_SAMPLES = TRAIN_SIZE + VAL_SIZE + TEST_SIZE


def preprocess_data():
    processed_data = {}

    # read the source and target language file line by line and do not include duplicate sentences
    counter = 0
    src_lang = []
    trg_lang = []
    seen = set()
    with open(src_lang_file, "r") as src_fh, open(trg_lang_file, "r") as trg_fh:
        for src_line, trg_line in zip(src_fh, trg_fh):
            if src_line not in seen:
                seen.add(src_line)
                src_lang.append(src_line)
                trg_lang.append(trg_line)
                counter += 1
                if counter >= MAX_SAMPLES:
                    break

    # encode the data
    encoded_src_lang, src_vocab = tokenize(src_lang)
    encoded_trg_lang, trg_vocab = tokenize(trg_lang)

    print(f"Source language vocabulary size: {len(src_vocab)}")
    print(f"Target language vocabulary size: {len(trg_vocab)}")

    # split the data into train, val and test sets
    train_data = tf.data.Dataset.from_tensor_slices(
        (encoded_src_lang[:TRAIN_SIZE], encoded_trg_lang[:TRAIN_SIZE]))
    train_data = train_data.shuffle(len(train_data), reshuffle_each_iteration=True).batch(
        BATCH_SIZE, drop_remainder=True)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    val_data = tf.data.Dataset.from_tensor_slices(
        (encoded_src_lang[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE], encoded_trg_lang[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]))
    val_data = val_data.shuffle(len(val_data), reshuffle_each_iteration=True).batch(
        BATCH_SIZE, drop_remainder=True)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_tensor_slices(
        (encoded_src_lang[TRAIN_SIZE+VAL_SIZE:], encoded_trg_lang[TRAIN_SIZE+VAL_SIZE:]))
    test_data = test_data.shuffle(len(test_data), reshuffle_each_iteration=True).batch(
        1, drop_remainder=True)
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

    processed_data["train"] = train_data
    processed_data["val"] = val_data
    processed_data["test"] = test_data
    processed_data["src_vocab"] = src_vocab
    processed_data["trg_vocab"] = trg_vocab

    return processed_data


def clean_tokens(sentence):
    """Cleans a sentence."""
    sentence = sentence.replace("'", "")
    sentence = re.sub(r"[^a-zA-Z]+", r" ", sentence)
    words = sentence.split()
    tokens = [word.lower().strip() for word in words]
    return tokens


def tokenize(corpus):
    """Returns an encoded representation of the corpus data"""
    vocabulary = ["pad", "SOS", "EOS"]
    word2idx = {
        "pad": 0,
        "SOS": 1,
        "EOS": 2,
    }
    idx2word = {
        0: "pad",
        1: "SOS",
        2: "EOS"
    }
    encoded_corpus = []
    counter = 3
    for sentence in corpus:
        encoded_sentence = [1]              # add SOS token
        tokens = clean_tokens(sentence)
        if len(tokens) > MAX_SENTENCE_LENGTH - 2:
            tokens = tokens[:MAX_SENTENCE_LENGTH-2]    # strip long sentences
        for token in tokens:
            if token not in vocabulary:
                vocabulary.append(token)
                word2idx[token] = counter
                idx2word[counter] = token
                counter += 1
            encoded_sentence.append(word2idx[token])
        encoded_sentence.append(2)          # add EOS token

        # add padding to small sentences
        if len(encoded_sentence) < MAX_SENTENCE_LENGTH:
            while len(encoded_sentence) != MAX_SENTENCE_LENGTH:
                encoded_sentence.append(0)

        encoded_corpus.append(encoded_sentence)

    return encoded_corpus, vocabulary

