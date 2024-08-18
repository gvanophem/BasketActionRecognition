from argparse import ArgumentParser
import math

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Conv1D, MaxPooling1D, Attention, MultiHeadAttention, LayerNormalization, Layer, Flatten, Reshape, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2

#from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

import tensorflow.keras.backend as K

import tensorflow as tf

import numpy as np
import os

from lstm_processing import sliding_labels, read_json, clean_labels, data_to_train, sort_sequence
from lstm_model import Transformer, create_model, create_model_cls

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--size", default=200, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--embed", default=512, type=int)
    args = parser.parse_args()

    path = os.path.join("..", "data")
    T=args.size
    timestep = 200
    WINDOW_SIZE = T // timestep
    TEST_DIRECTORY = "KS-FR-ROANNE"
    NUM_PLAYERS = 10
    NUM_REF = 3
    embedding_dim = args.embed
    thirdDense = 64
    SEQUENCE_LENGTH = WINDOW_SIZE*3*(NUM_PLAYERS+NUM_REF)

    class_names_list = np.array(['No Game', 'Half-court', 'Slow Transition', 'EOB', 'SOB', 'Free Throw', 'Timeout', 'Jump Ball', 'Normal Transition', 'Fast Transition', 'No Action'])
    class_names_dic = {label:num for num,label in enumerate(class_names_list)}
    count_action_types = {class_names_dic[label]:0 for label in class_names_list}
    num_classes = 11

    checkpoint_dir = "models/MLP_lstm_" + str(args.size)
    print(checkpoint_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    print(latest)

    if T == 200:
        model=create_model(WINDOW_SIZE, SEQUENCE_LENGTH, second_dense=embedding_dim, third_dense=thirdDense, last_dense=64, n=2)
    else:
        model = create_model_cls(WINDOW_SIZE, SEQUENCE_LENGTH+embedding_dim, second_dense=embedding_dim, third_dense=thirdDense, last_dense=64, n=2)   #mettre second_dense=embed_dim
    

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.load_weights(latest).expect_partial()

    model.summary()

    data_X = []
    data_y = []

    print(os.listdir(path))

    for game in os.listdir(os.path.join(path, TEST_DIRECTORY)):
        print(game)
        annotations = read_json(os.path.join(path, TEST_DIRECTORY, game, "annotations.json"))
        detections = read_json(os.path.join(path, TEST_DIRECTORY, game, "detections.json"))["detections_lists"]
        detections.sort(key=lambda x:x['timestamp'])
        actions_ts, actions_durations, actions_labels = clean_labels(annotations)
        windows = []
        labels = []
        for window, label in sliding_labels(actions_ts, actions_durations, actions_labels, detections, T):
            data_window = data_to_train(window, WINDOW_SIZE)
            #sorted_window = sort_sequence(data_window)
            sorted_window = data_window.reshape((1,WINDOW_SIZE*3*(NUM_PLAYERS+NUM_REF)))
            count_action_types[class_names_dic[label]] += 1

            data_X.append(sorted_window)
            data_y.append(class_names_dic[label])
    print(len(data_X))

    X = np.array(data_X)
    print(X.shape)
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    cls_tokens = tf.Variable(tf.random.normal([X_tensor.shape[0], 1, embedding_dim]), name='cls_tokens')

    print(X_tensor.shape)

    input_X = tf.concat([cls_tokens, X_tensor], axis=2)
    print(input_X.shape)
    y = to_categorical(data_y).astype(int)
    y = np.array(y)

    if T==200:
        predicted_labels = model.predict(X)
    else:
        predicted_labels = model.predict(input_X)

    pred = np.zeros(len(predicted_labels))
    for i in range(len(predicted_labels)):
        pred[i] = np.argmax(predicted_labels[i])
    
    with np.printoptions(threshold=np.inf):
        print(pred)

    test_labels = np.array(data_y)

    with np.printoptions(threshold=np.inf):
        print(test_labels)

    num_correct = np.sum(test_labels == pred)

    total_samples = len(test_labels)

    accuracy = (num_correct / total_samples) * 100

    #print(sum(predicted_labels[-1]))

    print("Accuracy of model " + str(latest) + " is " + str(accuracy))

    save_file = 'numpy_output/data_size{}_epochs{}_embed{}_thirdDense{}.npy'.format(str(T), str(args.epochs), str(embedding_dim), str(thirdDense))
    with open(save_file, 'wb') as f:
        np.save(f, test_labels)
        np.save(f, pred)
        np.save(f, predicted_labels)
