import tensorflow as tf
from tqdm import tqdm


def translate(samples, model, source_vocab, target_vocab):
    """
    Use the transformer model and translate sentences
    src_lang => German
    trg_lang => English
    """
    # Write the sentences to a file for analysis later
    fh = open("translated.txt", "a")

    for (encoder_inputs, targets) in tqdm(samples):
        # Set the decoder inputs
        decoder_inputs = targets[:, :-1]
        # Set the target outputs, right shifted
        decoder_outputs = targets[:, 1:]

        # Call the transformer and get the predicted output
        predictions = model(encoder_inputs, decoder_inputs, False)
        predictions = tf.squeeze(predictions)

        # greedy search
        predicted_ids = tf.cast(tf.argmax(predictions, axis=1), tf.int32)
        translated_sentence = " ".join([target_vocab[idx] for idx in predicted_ids[:-1]])

        # decode the original sentences
        originals = tf.squeeze(targets)
        original_idx = tf.cast(originals, tf.int32)
        target_sentence = " ".join([target_vocab[idx] for idx in original_idx[1:-1]])

        source_encoded = tf.squeeze(encoder_inputs)
        source_encoded_id = tf.cast(source_encoded, tf.int32)
        source_sentence = " ".join([source_vocab[idx] for idx in source_encoded_id[1:-1]])

        fh.write(f"Source Language Sentence: {source_sentence}\n")
        fh.write(f"Target Language Sentence: {target_sentence}\n")
        fh.write(f"Transformer model output: {translated_sentence}\n")
        fh.write("\n")

    fh.close()




