from preprocessing import preprocess_data
from model import Transformer
from train import train_val
from test import test
from translate import translate
from plot import plot_results
import tensorflow as tf

EPOCHS = 10
D_MODEL = 512
N_LAYERS = 6
FFN_X = 2048
N_HEADS = 8
DROPOUT_RATE = 0.1

# load the pre processed data
processed_data = preprocess_data()

# Clean the session
tf.keras.backend.clear_session()

# Create a transformer model
transformer = Transformer(
    encoder_vocab_size=len(processed_data["src_vocab"]),
    decoder_vocab_size=len(processed_data["trg_vocab"]),
    d_model=D_MODEL,
    num_layers=N_LAYERS,
    FFN_x=FFN_X,
    num_heads=N_HEADS,
    dropout_rate=DROPOUT_RATE
)

# Train the model
train_losses, train_accuracies, val_losses, val_accuracies = train_val(
    model=transformer, train_data=processed_data["train"],
    val_data=processed_data["val"], epoch=EPOCHS
)

# Plot training and validation loss and accuracy
plot_results(train_losses, train_accuracies, val_losses, val_accuracies, EPOCHS)


# Test the model
test(dataset=processed_data["test"], transformer=transformer)

# Translate some sentences
translate(
    samples=processed_data["test"],
    model=transformer,
    source_vocab=processed_data["src_vocab"],
    target_vocab=processed_data["trg_vocab"]
)


