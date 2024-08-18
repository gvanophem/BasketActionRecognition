from tensorflow.keras.layers import Dense, Input, Dropout, LayerNormalization, TimeDistributed, Reshape, LSTM, RNN, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2

import tensorflow as tf

import math

def create_mlp(input_shape, first_units, second_units):
    mlp_input = Input(shape=input_shape)
    dense_1 = Dense(first_units, activation='relu')(mlp_input)
    dense_2 = Dense(second_units, activation='relu')(dense_1)
    return Model(inputs = mlp_input, outputs=dense_2)


def gelu(x):
    #Original paper: https://arxiv.org/abs/1606.08415
    return x * 0.5 * (1.0 + tf.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x,3)))))    #this is gelu
    #return max(0,x)     

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class Transformer(tf.keras.layers.Layer):
    # Code for this transformer inspired from https://www.geeksforgeeks.org/transformer-model-from-scratch-using-tensorflow/
    # Also required a little bit of help from ChatGPT to adapt it
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.4):
        super(Transformer, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=gelu, kernel_regularizer=L2(0.01)),
                Dropout(dropout),
                Dense(embed_dim),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1
    
def create_model(win_size, seq_len, num_players=10, num_referees=3, first_dense=64, second_dense=32, third_dense=64, last_dense=64, num_classes=11, num_heads=4, ff_dim=64, n=1):
    input_layer = Input(shape=(win_size, seq_len))

    reshaped_input = Reshape((num_players+num_referees, 3))(input_layer)

    tokens_1 = Dense(first_dense, activation='relu')(reshaped_input)
    tokens_2 = Dense(second_dense, activation='relu')(tokens_1)

    embed_dim = tokens_2.shape[-1]  # Embedding dimension 

    transformer = Transformer(embed_dim, num_heads, ff_dim)

    transformed_tokens = transformer(tokens_2)

    trans_dim = transformed_tokens.shape[-1]

    dense_transformed = Dense(third_dense, activation='relu')(transformed_tokens)

    reshaped_transformed = Reshape((win_size, third_dense*(num_players+num_referees)))(dense_transformed)

    lstm_layer = LSTM(n * 128, return_sequences=True)(reshaped_transformed)
    lstm_layer_1 = LSTM(n * 256, return_sequences=True)(lstm_layer)
    lstm_layer_2 = LSTM(n * 128, return_sequences=False)(lstm_layer_1)

    dense_layer_1 = Dense(last_dense, activation='relu')(lstm_layer_2)

    reshaped_before_output = Reshape((win_size, last_dense))(dense_layer_1)

    dense_layer_2 = Dense(num_classes, activation='softmax')(reshaped_before_output)

    output = Flatten()(dense_layer_2)

    model = Model(inputs=input_layer, outputs=output)
    return model

def create_model_cls(win_size, seq_len, num_players=10, num_referees=3, first_dense=64, second_dense=32, third_dense=64, last_dense=64, num_classes=11, num_heads=4, ff_dim=64, n=1):

    input_layer = Input(shape=(1, seq_len))

    cls_token = input_layer[:,:,:second_dense]
    input_tensor = input_layer[:,:,second_dense:]

    reshaped_input = Reshape((win_size, (num_players+num_referees), 3))(input_tensor)

    tokens_1 = Dense(first_dense, activation='relu')(reshaped_input)
    tokens_2 = Dense(second_dense, activation='relu')(tokens_1)

    reshaped_tokens = Reshape((win_size*(num_players+num_referees), second_dense))(tokens_2)
    transformer_input = tf.concat([cls_token, reshaped_tokens], axis=1)

    embed_dim = tokens_2.shape[-1]  # Embedding dimension 

    transformer = Transformer(embed_dim, num_heads, ff_dim)

    transformed_tokens = transformer(transformer_input)

    trans_dim = transformed_tokens.shape[-1]

    dense_transformed = Dense(third_dense, activation='relu')(transformed_tokens)

    cls_token, detection_transformed = tf.split(dense_transformed, [1, win_size*(num_players+num_referees)], axis=1)

    lstm_layer = LSTM(third_dense, return_sequences=True)(cls_token)
    lstm_layer_1 = LSTM(4*third_dense, return_sequences=True)(lstm_layer)
    lstm_layer_2 = LSTM(2*third_dense, return_sequences=False)(lstm_layer_1)


    dense_layer_1 = Dense(last_dense, activation='relu')(lstm_layer_2)


    dense_layer_2 = Dense(num_classes, activation='softmax')(dense_layer_1)

    output = Flatten()(dense_layer_2)

    model = Model(inputs=input_layer, outputs=dense_layer_2)
    return model