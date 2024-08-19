import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from image_processing import read_json, clean_labels, sliding_labels, create_image


#GLOBAL PARAMETERS
T = 4000
TEST_DIRECTORY = "KS-FR-ROANNE"
PATH = "../data/"
print(tf.__version__)
#tf.debugging.set_log_device_placement(True)
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "AVAILABLE" if gpu else "NOT AVAILABLE")


path = "../data/"

height = 240
width = 320
channels = 4  # Assuming RGB images  maybe 3
input_shape = (height, width, 1)


#RETRIEVING OF THE FITTED MODEL
model_name = "models/model_arrow_50.keras"

model = tf.keras.models.load_model(model_name)

predicted_labels = []
test_labels = []

num_classes = 11


class_names_list = np.array(['No Game', 'Half-court', 'Slow Transition', 'EOB', 'SOB', 'Free Throw', 'Timeout', 'Jump Ball', 'Normal Transition', 'Fast Transition', 'No Action'])
class_names_dic = {'No Game' : 0, 'Half-court' : 1, 'Slow Transition' : 2, 'EOB' : 3, 'SOB' : 4, 'Free Throw' : 5, 'Timeout' : 6, 'Jump Ball':7, 'Normal Transition':8, 'Fast Transition':9, 'No Action':10}


#PREDICT THE CLASSES OF EACH WINDOW

for game in os.listdir(os.path.join(path, TEST_DIRECTORY)):
    annotations = read_json(os.path.join(path, TEST_DIRECTORY, game, "annotations.json"))
    detections = read_json(os.path.join(path, TEST_DIRECTORY, game, "detections.json"))["detections_lists"]
    detections.sort(key=lambda x:x["timestamp"])
    actions_ts, actions_durations, actions_labels = clean_labels(annotations)
    windows = []
    labels = []
    for window, label in sliding_labels(actions_ts, actions_durations, actions_labels, detections, T):
        windows.append(create_image(window))
        labels.append(label)
    windows = np.array(windows)
    windows = windows / 255.0
    windows = windows.reshape(len(windows), height, width, 1)

    test_labels = np.zeros(len(labels))
    for i in range(len(test_labels)):
        test_labels[i] = int(class_names_dic[labels[i]])

    predicted_labels = model.predict(windows)

pred = np.zeros(len(predicted_labels))
for i in range(len(predicted_labels)):
    pred[i] = np.argmax(predicted_labels[i])


#COMPARE THE TRUE CLASSES WITH THE PREDICTED ONES

num_correct = np.sum(test_labels == pred)

total_samples = len(test_labels)

accuracy = (num_correct / total_samples) * 100

print("Accuracy of model " + str(model_name) + " is " + str(accuracy))


#SAVE THE RESULTS ON THE TEST SET

save_file = 'output_CNN/data_size{}_epochs{}.npy'.format(str(T), str(50))
with open(save_file, 'wb') as f:
    np.save(f, test_labels)
    np.save(f, pred)
    np.save(f, predicted_labels)
