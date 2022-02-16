import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def single_attention(Q, K, V, mask):
    """
    Calculate attention using the formula:
    Attention(Q, K, V) = Softmax(QK^T / squareroot(d_k)) V
    Q => (64, 8, 15, 64)
    K => (64, 8, 15, 64)
    V => (64, 8, 15, 64)
    mask => (64, 1, 1, 15)
    """
    d_k = tf.cast(tf.shape(K)[-1], tf.float32)      # 64
    P = tf.matmul(Q, K, transpose_b=True)           # (64, 8, 15, 15)
    # Apply the scale factor to the dot product
    P = P / tf.math.sqrt(d_k)
    if mask is not None:
        P += (mask * -1e9)
    attention = tf.matmul(tf.nn.softmax(P, axis=-1), V)     # (64, 8, 15, 64)

    return attention


class MultiHeadAttention(layers.Layer):
    """
    MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # the number of heads
        self.d_model = None         # dimension of model
        self.dim_head = None        # dimension of each head
        self.Q_linear = None        # the Q matrix
        self.K_linear = None        # the K matrix
        self.V_linear = None        # the V matrix
        self.w0 = None

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.num_heads == 0
        self.dim_head = self.d_model // self.num_heads
        self.Q_linear = layers.Dense(units=self.d_model)
        self.K_linear = layers.Dense(units=self.d_model)
        self.V_linear = layers.Dense(units=self.d_model)
        self.w0 = layers.Dense(units=self.d_model)

    def projections(self, inputs, batch_size):
        # inputs are of the shape:  (batch_size, seq_length, d_model)
        shape = (batch_size, -1, self.num_heads, self.dim_head)         # shape of the projections
        splits = tf.reshape(inputs, shape=shape)
        return tf.transpose(splits, perm=[0, 2, 1, 3])  # (batch_size, nb_proj, seq_length, d_proj)

    def call(self, Q, K, V, mask):
        """
        Q => (64, 15, 512)
        K => (64, 15, 512)
        V => (64, 15, 512)
        mask => (64, 1, 1, 15)
        """
        batch_size = tf.shape(Q)[0]
        Q = self.Q_linear(Q)                    # (64, 15, 512)
        Q = self.projections(Q, batch_size)     # (64, 8, 15, 64)
        K = self.K_linear(K)                    # (64, 15, 512)
        K = self.projections(K, batch_size)     # (64, 8, 15, 64)
        V = self.V_linear(V)                    # (64, 15, 512)
        V = self.projections(V, batch_size)     # (64, 8, 15, 64)
        attention = single_attention(Q, K, V, mask)     # (64, 8, 15, 64)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])      # (64, 15, 8, 64)
        all_attentions = tf.reshape(attention, shape=(batch_size, -1, self.d_model))       # (64, 15, 512)
        outputs = self.w0(all_attentions)        # (64, 15, 512)

        return outputs


class PositionalEncoding(layers.Layer):
    """
    PE(pos,2i) = sin(pos/10000^2i/d_model)
    PE(pos,2i+1) = cos(pos/10000^2i/d_model)
    """
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def call(self, inputs):
        # inputs => shape(64, 15, 512)
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        pos = np.arange(seq_length)[:, np.newaxis]      # (15, 1)
        i = np.arange(d_model)[np.newaxis, :]           # (1, 512)
        power = (2 * (i // 2)) / np.float32(d_model)    # (1, 512)
        angle = pos / (np.power(10000.0, power))        # (15, 512)
        angle[:, 0::2] = np.sin(angle[:, 0::2])         # (15, 256)
        angle[:, 1::2] = np.cos(angle[:, 1::2])         # (15, 256)
        positional_encoding = angle[np.newaxis, ...]    # (1, 15, 512)

        return inputs + tf.cast(positional_encoding, tf.float32)


class EncoderLayer(layers.Layer):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, FFN_x, num_heads, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.FFN_x = FFN_x
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape = (64, 15, 512)
        self.d_model = input_shape[-1]
        self.ffn1 = layers.Dense(units=self.FFN_x, activation="relu")
        self.ffn2 = layers.Dense(units=self.d_model)
        self.multi_head_attention = MultiHeadAttention(self.num_heads)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, mask, training):
        # inputs => (64, 15, 512)
        # mask => (64, 1, 1, 15)
        attention = self.multi_head_attention(inputs, inputs, inputs, mask)     # (64, 15, 512)
        attention = self.dropout_1(attention, training=training)     # (64, 15, 512)
        attention = self.norm_1(attention + inputs)     # (64, 15, 512)
        outputs = self.ffn1(attention)     # (64, 15, 2048)
        outputs = self.ffn2(outputs)       # (64, 15, 512)
        outputs = self.dropout_2(outputs, training=training)       # (64, 15, 512)
        outputs = self.norm_2(outputs + attention)     # (64, 15, 512)

        return outputs


class Encoder(layers.Layer):
    """
    vocab_size = 18936
    d_model = 512
    num_layers = 6
    FFN_x = 2048
    num_heads = 8
    dropout_rate = 0.1
    """
    def __init__(self, num_layers, FFN_x, num_heads, dropout_rate, vocab_size, d_model, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.encoder_layers = [EncoderLayer(FFN_x, num_heads, dropout_rate) for _ in range(num_layers)]

    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs)    # (64, 15, 512)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.positional_encoding(outputs)     # (64, 15, 512)
        outputs = self.dropout(outputs, training)
        for i in range(self.num_layers):
            outputs = self.encoder_layers[i](outputs, mask, training)       # (64, 15, 512)

        return outputs


class DecoderLayer(layers.Layer):
    def __init__(self, FFN_x, num_heads, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_x = FFN_x
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.ffn1 = layers.Dense(units=self.FFN_x, activation="relu")
        self.ffn2 = layers.Dense(units=self.d_model)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)
        self.multi_head_attention_1 = MultiHeadAttention(self.num_heads)
        self.multi_head_attention_2 = MultiHeadAttention(self.num_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, encoded_outputs, mask_1, mask_2, training):
        """
        inputs => (64, 14, 512)
        encoded_outputs => (64, 15, 512)
        mask_1 => (64, 1, 14, 14)
        mask_2 => (64, 1, 1, 15)
        """
        attention_1 = self.multi_head_attention_1(inputs, inputs, inputs, mask_1)       # (64, 14, 512)
        attention_1 = self.dropout_1(attention_1, training)
        attention_1 = self.norm_1(attention_1 + inputs)     # (64, 14, 512)

        attention_2 = self.multi_head_attention_2(attention_1, encoded_outputs, encoded_outputs, mask_2)    # (64, 14, 512)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_2 + attention_1)        # (64, 14, 512)

        outputs = self.ffn1(attention_2)        # (64, 14, 2048)
        outputs = self.ffn2(outputs)            # (64, 14, 512)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs + attention_2)        # (64, 14, 512)

        return outputs


class Decoder(layers.Layer):
    """
    vocab_size = 13736
    d_model = 512
    num_layers = 6
    FFN_x = 2048
    num_heads = 8
    dropout_rate = 0.1
    """
    def __init__(self, num_layers, FFN_x, num_heads, dropout_rate, vocab_size, d_model, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.decoder_layers = [DecoderLayer(FFN_x, num_heads, dropout_rate) for _ in range(num_layers)]

    def call(self, inputs, encoded_outputs, mask_1, mask_2, training):
        """
        inputs => (64, 14)
        encoded_outputs => (64, 15, 512)
        mask_1 => (64, 1, 14, 14)
        mask_2 => (64, 1, 1, 15)
        """
        outputs = self.embedding(inputs)        # (64, 14, 512)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))      # (64, 14, 512)
        outputs = self.positional_encoding(outputs)     # (64, 14, 512)
        outputs = self.dropout(outputs, training)
        for i in range(self.num_layers):
            outputs = self.decoder_layers[i](outputs, encoded_outputs, mask_1, mask_2, training)    # (64, 14, 512)

        return outputs


class Transformer(tf.keras.Model):
    """
    encoder_vocab_size = 18936
    decoder_vocab_size = 13736
    d_model = 512
    num_layers = 6
    FFN_x = 2048
    num_heads = 8
    dropout_rate = 0.1
    """
    def __init__(self, encoder_vocab_size, decoder_vocab_size, d_model, num_layers, FFN_x, num_heads, dropout_rate, name="transformer"):
        super(Transformer, self).__init__(name=name)
        self.encoder = Encoder(num_layers, FFN_x, num_heads, dropout_rate, encoder_vocab_size, d_model)
        self.decoder = Decoder(num_layers, FFN_x, num_heads, dropout_rate, decoder_vocab_size, d_model)
        self.linear = layers.Dense(units=decoder_vocab_size, name="linear")

    def mask_padding(self, sequence):
        mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)      # a new tensor of all zeros
        return mask[:, tf.newaxis, tf.newaxis, :]                   # add 2 new dimensions in between

    def mask_attention_1(self, sequence):
        sequence_len = tf.shape(sequence)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((sequence_len, sequence_len)), -1, 0) # 14x14 matrix with lower triangle to 0
        return mask

    def call(self, encoder_inputs, decoder_inputs, training):
        """
        encoder_inputs: shape => (64, 15)
        decoder_inputs: shape => (64, 14)
        """
        encoder_mask = self.mask_padding(encoder_inputs)        # (64, 1, 1, 15)
        decoder_mask_1 = tf.maximum(                            # (64, 1, 1, 14)
            self.mask_padding(decoder_inputs),
            self.mask_attention_1(decoder_inputs)
        )
        decoder_mask_2 = self.mask_padding(encoder_inputs)      # (64, 1, 1, 15)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask, training)  # (64, 15, 512)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, decoder_mask_1, decoder_mask_2, training)   # (64, 14, 512)
        return self.linear(decoder_outputs)     # (64, 14, 13736)

