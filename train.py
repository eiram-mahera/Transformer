import tensorflow as tf
from tqdm import tqdm

D_MODEL = 512
checkpoint_path = "model_checkpoint"


def loss_function(loss_object, target, predicted):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss = loss_object(target, predicted)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    # Scheduler for learning rate decay
    def __init__(self, d_model, warmup_steps=4000):
        super(LRScheduler, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_val(model, train_data, val_data, epoch):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    learning_rate = LRScheduler(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    checkpoint = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Restored checkpoint")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for ep in range(epoch):
        print(f"Epoch {ep+1}")
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # train the model
        # encoder_inputs => (64, 15)
        # targets => (64, 15)
        for (encoder_inputs, targets) in tqdm(train_data):
            decoder_inputs = targets[:, :-1]    # remove the last token before feeding to the transformer
            decoder_outputs = targets[:, 1:]    # remove the first token which is SOS

            with tf.GradientTape() as tape:
                predictions = model(encoder_inputs, decoder_inputs, True)       # (64, 14, 13733)
                loss = loss_function(loss_object, decoder_outputs, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(decoder_outputs, predictions)

        train_losses.append(train_loss.result())
        train_accuracies.append(train_accuracy.result())
        print("Training - Loss: {:.4f} Accuracy: {:.2f}".format(train_loss.result(), train_accuracy.result()))

        cs = checkpoint_manager.save()
        print(f"Checkpoint saved in {cs}")

        # evaluate the model
        for (encoder_inputs, targets) in tqdm(val_data):
            decoder_inputs = targets[:, :-1]
            decoder_outputs = targets[:, 1:]

            predictions = model(encoder_inputs, decoder_inputs, True)
            loss = loss_function(loss_object, decoder_outputs, predictions)

            val_loss(loss)
            val_accuracy(decoder_outputs, predictions)

        val_losses.append(train_loss.result())
        val_accuracies.append(train_accuracy.result())
        print("Validation - Loss: {:.4f} Accuracy: {:.2f}".format(val_loss.result(), val_accuracy.result()))

    return train_losses, train_accuracies, val_losses, val_accuracies


