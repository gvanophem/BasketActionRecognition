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
    parser.add_argument("--size", default=400, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--embed", default=32, type=int)
    args = parser.parse_args()

    log_dir = os.path.join('Logs_LSTM')
    tb_callback = TensorBoard(log_dir=log_dir)

    early_stopping = EarlyStopping(monitor="val_categorical_accuracy", patience=50, restore_best_weights=True)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss'
                                            , patience = 50
                                            , verbose=1
                                            ,factor=0.75
                                            , min_lr=0.00001)

    path = os.path.join("..","data")
    T=args.size
    timestep = 200
    WINDOW_SIZE = T // timestep
    TEST_DIRECTORY = "KS-FR-ROANNE"
    NUM_PLAYERS = 10
    NUM_REF = 3
    embedding_dim = args.embed
    SEQUENCE_LENGTH = WINDOW_SIZE*3*(NUM_PLAYERS+NUM_REF)

    class_names_list = np.array(['No Game', 'Half-court', 'Slow Transition', 'EOB', 'SOB', 'Free Throw', 'Timeout', 'Jump Ball', 'Normal Transition', 'Fast Transition', 'No Action'])
    class_names_dic = {label:num for num,label in enumerate(class_names_list)}
    count_action_types = {class_names_dic[label]:0 for label in class_names_list}
    num_classes = 11

    data_X = []
    data_y = []

    for field in os.listdir(path):
        print(field)
        if(field != TEST_DIRECTORY):
            for game in os.listdir(os.path.join(path, field)):
                print(game)
                annotations = read_json(os.path.join(path, field, game, "annotations.json"))
                detections = read_json(os.path.join(path, field, game, "detections.json"))["detections_lists"]
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



    
    if T == 200:
        model=create_model(WINDOW_SIZE, SEQUENCE_LENGTH, second_dense=embedding_dim, third_dense=64, last_dense=64, n=2)
    else:
        model = create_model_cls(WINDOW_SIZE, SEQUENCE_LENGTH+embedding_dim, second_dense=embedding_dim, third_dense=64, last_dense=64, n=2)   #mettre second_dense=embed_dim
    model.summary()

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    num_labels = X.shape[0]
    val_length = num_labels//5

    class_weights = {class_int:num_labels/(num_classes*count_label) for class_int,count_label in count_action_types.items()}

    print(class_weights)

    print(count_action_types)

    #With VALIDATION
    #model.fit(X[:-val_length], y[:-val_length], epochs=args.epochs, validation_data=(X[-val_length:], y[-val_length:]), callbacks=[tb_callback], batch_size=128, class_weight=class_weights)
    
    #NO VALIDATION
    print(input_X.shape)
    print(y.shape)
    if T==200:
        model.fit(X, y, epochs=args.epochs, callbacks=[tb_callback], batch_size=args.batch_size, class_weight=class_weights)
    else :
        model.fit(input_X, y, epochs=args.epochs, callbacks=[tb_callback], batch_size=args.batch_size, class_weight=class_weights)

    model_path = 'models/MLP_lstm_{}/transformer_{}'.format(args.size, args.epochs)
    print("Saving in : " + model_path)
    print(os.path.exists(model_path))
    if os.path.exists(model_path):
        os.remove(model_path)
    print(os.path.exists(model_path))
    model.save_weights(model_path)
    #model.save_weights('models/lstm/attention_{}'.format(args.size))