import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def read_json(filename):
     """ Method that reads the data available in a filename of type json and returns this data

    Parameters
    ----------
    filename : str
        The name of the file you want to read

    Raises
    ------
    ValueError : the file is not of type .json

    Returns
    -------
    The data in the json file in a dictionary
    """
     if not filename.endswith(".json"):
         raise ValueError("File should be of type .json")
     with open(filename, 'r') as file:
         data = json.load(file)
     return data

def create_image(window):
    """This method creates an image based on a window

    Parameters
    ----------
    window : list
        List of detections that needs to be transformed into an image    

    Returns
    -------
    An array representing the image
    """
    size = 1
    cmap = plt.get_cmap('gray')
    plt.xlim(0,2800)
    plt.ylim(0,1500)
    plt.gca().set_axis_off()
    for detection in window : 
        ref_cnt = 0
        game = detection['detections']
        x = np.zeros(len(game))
        y = np.zeros(len(game))
        color = cmap((1/2) - ((size) / (2*len(window) )))
        arrow_length = 30*size/len(window)
        for i in range (len(game)):
            position = game[i]
            if(position['label'] == "player"):
                x[i-ref_cnt] = position['pos_feet'][0]
                y[i-ref_cnt] = position['pos_feet'][1]
                angle = np.deg2rad(position['orientation'])
                dx = arrow_length * np.cos(angle)
                dy = arrow_length * np.sin(angle)
                plt.arrow(x[i-ref_cnt], y[i-ref_cnt], dy, -dx, color=color, head_width=arrow_length/5, head_length=arrow_length/5)
            else : 
                ref_cnt+=1
        plt.scatter(x[:len(game)-ref_cnt],y[:len(game)-ref_cnt],s=20*size/len(window), color = color)
        size+=1
    canvas = plt.gcf().canvas
    canvas.draw()
    img = np.array(canvas.buffer_rgba())
    img = np.rint(img[...,:3] @ [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    plt.close()

    new_height, new_width = 240, 320

    resized_image = resize(img, (new_height, new_width), anti_aliasing=True)

    return resized_image

def find_detection_index(timestamp, detections, index):
    i = 0
    len_detection = len(detections)
    while index + i < len_detection:
        if detections[index+i]['timestamp'] >= timestamp:
            return index + i
        i+= 1
    return -1

def sliding_labels(actions_ts, actions_durations, actions_labels, detections, size):
    i = 0
    action_index = 0
    detection_len = len(detections)
    action_len = len(actions_ts)
    print("Action len : " + str(action_len))
    while i < detection_len :
        start_ts = detections[i]['timestamp']
        mid = start_ts + (size // 2)
        j = 0
        while action_index + j < action_len:
            if(actions_ts[action_index+j] > mid):
                break
            j+= 1
        j-= 1
        action_index = max(0,action_index + j)


        window = []
        j = 0
        window = []
        while i+j < detection_len and (detections[i+j]['timestamp'] - detections[i]['timestamp']) < size:
            window.append(detections[i+j])
            j+= 1

        if actions_ts[action_index] > mid : 
            i = find_detection_index((actions_ts[action_index] - (size // 2)), detections, i)
        elif actions_durations[action_index] < mid - actions_ts[action_index] : 
            if action_index < action_len - 1:
                i = find_detection_index((actions_ts[action_index + 1] - (size // 2)), detections, i)
                yield window, "No Action"
            else:
                break
        else :
            i = find_detection_index(start_ts + (size//4), detections, i)
            yield window, actions_labels[action_index]

        if i == -1:
           break

def clean_labels(annotations):
  #remove slow transitions
  actions_ts = np.array(annotations['actions_ts'])
  actions_durations = np.array(annotations['actions_durations'])
  actions_labels = np.array(annotations['actions_labels'])
  i = 0
  flag = 0
  to_delete = []
  while i < len(actions_ts):
    if actions_labels[i] == "Slow Transition" or actions_labels[i] == "Fast Transition" or actions_labels[i] == "Normal Transition":

      if i==len(actions_ts)-1:
        #Quand transition est la dernière action du match
        flag+=1
        to_delete.append(i)
        if flag != 0:
          i+=1
          duration = actions_ts[i-1] - actions_ts[i-flag]
          actions_ts = np.delete(actions_ts, to_delete)
          actions_labels = np.delete(actions_labels, to_delete)
          actions_durations = np.delete(actions_durations, to_delete)
          actions_durations[i - flag] = duration
          to_delete = []
          i-= flag-1
          flag = 0
      elif flag != 0:
        flag+=1
        to_delete.append(i)
      else :
        flag = 1
    else :
      if flag != 0:
        duration = actions_ts[i-1] - actions_ts[i-flag]
        actions_ts = np.delete(actions_ts, to_delete)
        actions_labels = np.delete(actions_labels, to_delete)
        actions_durations = np.delete(actions_durations, to_delete)
        actions_durations[i - flag ] = duration
        to_delete = []
        i-= flag-1
        flag = 0
    i+= 1
  return actions_ts, actions_durations, actions_labels
