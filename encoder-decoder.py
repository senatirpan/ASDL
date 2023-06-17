# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np

# %% [markdown]
# Define encoder and decoder layers:

# %%
class EncoderLayer(keras.layers.Layer):
    def __init__(self, emb_dim, num_heads, hid_dim, dropout):
        super(EncoderLayer, self).__init__()

        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = keras.layers.Dense(hid_dim, activation='relu')
        self.dense2 = keras.layers.Dense(emb_dim)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.norm1(inputs + attention_output)

        intermediate_output = self.dense1(attention_output)
        intermediate_output = self.dense2(intermediate_output)
        intermediate_output = self.dropout2(intermediate_output, training=training)
        intermediate_output = self.norm2(attention_output + intermediate_output)

        return intermediate_output


class DecoderLayer(keras.layers.Layer):
    def __init__(self, emb_dim, num_heads, hid_dim, dropout):
        super(DecoderLayer, self).__init__()

        self.attention1 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.attention2 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = keras.layers.Dense(hid_dim, activation='relu')
        self.dense2 = keras.layers.Dense(emb_dim)
        self.dropout3 = keras.layers.Dropout(dropout)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, encoder_outputs, training=True):
        attention_output1 = self.attention1(inputs, inputs)
        attention_output1 = self.dropout1(attention_output1, training=training)
        attention_output1 = self.norm1(inputs + attention_output1)

        attention_output2 = self.attention2(attention_output1, encoder_outputs)
        attention_output2 = self.dropout2(attention_output2, training=training)
        attention_output2 = self.norm2(attention_output1 + attention_output2)

        intermediate_output = self.dense1(attention_output2)
        intermediate_output = self.dense2(intermediate_output)
        intermediate_output = self.dropout3(intermediate_output, training=training)
        intermediate_output = self.norm3(attention_output2 + intermediate_output)

        return intermediate_output


# %% [markdown]
# Define encoder and decoder models:

# %%
class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, emb_dim, num_heads, hid_dim, input_vocab_size, dropout):
        super(Encoder, self).__init__()

        self.emb_dim = emb_dim
        self.embedding = keras.layers.Embedding(input_vocab_size, emb_dim)
        self.dropout = keras.layers.Dropout(dropout)
        self.encoder_layers = [EncoderLayer(emb_dim, num_heads, hid_dim, dropout) for _ in range(num_layers)]

    def call(self, inputs, training=True):
        inputs = self.embedding(inputs)
        inputs *= tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))
        inputs = self.dropout(inputs, training=training)

        for encoder_layer in self.encoder_layers:
            inputs = encoder_layer(inputs, training=training)

        return inputs


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, emb_dim, num_heads, hid_dim, output_vocab_size, dropout):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.embedding = keras.layers.Embedding(output_vocab_size, emb_dim)
        self.dropout = keras.layers.Dropout(dropout)
        self.decoder_layers = [DecoderLayer(emb_dim, num_heads, hid_dim, dropout) for _ in range(num_layers)]

    def call(self, inputs, encoder_outputs, training=True):
        inputs = self.embedding(inputs)
        inputs *= tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))
        inputs = self.dropout(inputs, training=training)

        for decoder_layer in self.decoder_layers:
            inputs = decoder_layer(inputs, encoder_outputs, training=training)

        return inputs


# %% [markdown]
# Define the EncoderDecoder model:

# %%

class EncoderDecoder(keras.Model):
    def __init__(self, num_layers, emb_dim, num_heads, hid_dim, input_vocab_size, output_vocab_size, dropout):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(num_layers, emb_dim, num_heads, hid_dim, input_vocab_size, dropout)
        self.decoder = Decoder(num_layers, emb_dim, num_heads, hid_dim, output_vocab_size, dropout)
        self.final_dense = keras.layers.Dense(output_vocab_size)

    def call(self, inputs, training=True):
        encoder_outputs = self.encoder(inputs, training=training)
        decoder_outputs = self.decoder(target_data, encoder_outputs, training=training)  # Use inputs instead of targets
        final_outputs = self.final_dense(decoder_outputs)
        return final_outputs



# %% [markdown]
# load train.json data:

# %%
import json
train_data_path = "/Users/sena/Desktop/sena-personal/sena-repos/ASDL/train_data/train.json"

input_data = []
target_data = []

with open(train_data_path, 'r') as file:
    for line in file:
        example = json.loads(line)
        input_data.append(example['nl'])
        target_data.append(example['code'])

# check the label is nparrray:
print(type(input_data))

# convert to numpy array
input_data = np.asarray(input_data)
target_data = np.asarray(target_data)

print(type(input_data))



# %%
max_length = 0

with open(train_data_path, 'r') as file:
    for line in file:
        example = json.loads(line)
        input_sequence = example['nl']
        sequence_length = len(input_sequence.split())  # Split the sequence into tokens and get the length
        max_length = max(max_length, sequence_length)

print("Maximum sequence length:", max_length)


# %%
# Preprocess and encode the input data
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_data)
encoded_input_data = tokenizer.texts_to_sequences(input_data)
padded_input_data = pad_sequences(encoded_input_data, maxlen=max_length)


# %%
# Define model hyperparameters
num_layers = 2
emb_dim = 32
num_heads = 4
hid_dim = 64
input_vocab_size = 100
output_vocab_size = 100
dropout = 0.1

# Create an instance of the EncoderDecoder model
model = EncoderDecoder(num_layers, emb_dim, num_heads, hid_dim, input_vocab_size, output_vocab_size, dropout)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(padded_input_data, target_data, epochs=10, batch_size=2)


