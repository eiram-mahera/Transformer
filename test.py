import tensorflow as tf
from tqdm import tqdm


def test(dataset, transformer):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    for (encoder_inputs, targets) in tqdm(dataset):
        # Set the decoder inputs
        decoder_inputs = targets[:, :-1]
        # Set the target outputs, right shifted
        decoder_outputs = targets[:, 1:]
        # Call the transformer and get the predicted output
        predictions = transformer(encoder_inputs, decoder_inputs, False)
        test_accuracy(decoder_outputs, predictions)

    print(f"Test accuracy is {test_accuracy.result()}")




