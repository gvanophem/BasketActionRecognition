import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from image_processing import read_json, clean_labels, sliding_labels, create_image

T = 4000
TEST_DIRECTORY = "KS-FR-ROANNE"

path = "../data/"

height = 240
width = 320
input_shape = (height, width, 1)

dr = 1

num_classes = 11

model = models.Sequential()

model.add(layers.Input(shape=input_shape))

model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu', dilation_rate=dr))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32, (3,3), activation='relu', dilation_rate=dr))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class_names_list = np.array(['No Game', 'Half-court', 'Slow Transition', 'EOB', 'SOB', 'Free Throw', 'Timeout', 'Jump Ball', 'Normal Transition', 'Fast Transition', 'No Action'])
class_names_dic = {'No Game' : 0, 'Half-court' : 1, 'Slow Transition' : 2, 'EOB' : 3, 'SOB' : 4, 'Free Throw' : 5, 'Timeout' : 6, 'Jump Ball':7, 'Normal Transition':8, 'Fast Transition':9, 'No Action':10}


def train_model(root_dir):
  for field in os.listdir(root_dir):
    print(field)
    if(field != "KS-FR-STCHAMOND"):
      for game in os.listdir(os.path.join(root_dir, field)):
        count_action_types = {class_names_dic[label]:0 for label in class_names_list}
        print(game)
        annotations = read_json(os.path.join(root_dir, field, game, "annotations.json"))
        detections = read_json(os.path.join(root_dir, field, game, "detections.json"))["detections_lists"]
        detections.sort(key=lambda x:x['timestamp'])
        actions_ts, actions_durations, actions_labels = clean_labels(annotations)
        windows = []
        labels = []
        for window, label in sliding_labels(actions_ts, actions_durations, actions_labels, detections, T):
            windows.append(create_image(window))
            labels.append(label)
        windows = np.array(windows)
        windows = windows / 255.0
        windows = windows.reshape(len(windows), height, width, 1)

        train_labels = np.zeros(len(labels))
        for i in range(len(train_labels)):
          train_labels[i] = int(class_names_dic[labels[i]])
          count_action_types[class_names_dic[labels[i]]] += 1

        num_labels = windows.shape[0]
        class_weights = {class_int:num_labels/(num_classes*count_label+1) for class_int,count_label in count_action_types.items()}


        #print(train_labels)
        length_validation = len(train_labels)//10
        print(windows.shape)
        model.fit(windows, train_labels, class_weight=class_weights, epochs=50,batch_size=32, verbose=2)

train_model(path)

model.save("models/model_arrow_50.keras")