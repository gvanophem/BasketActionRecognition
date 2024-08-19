import json
import numpy as np

NUM_PLAYERS = 10


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

def find_detection_index(timestamp, detections, index):
    """This method finds the next detection which timestamp is greater than the timestamp given in parameter

    Parameters
    ----------
    timestamp : int
        Timestamp of the next detection we want to find

    detections : list
        List of all the player detections

    index : int
        index from which we need to start looking for the next detection

    Returns
    -------
    The index of the found detection, -1 if this detection is not possible to find.
    """
    i = 0
    len_detection = len(detections)
    while index + i < len_detection:
        if detections[index+i]['timestamp'] >= timestamp:
            return index + i
        i+= 1
    return -1

def sliding_labels(actions_ts, actions_durations, actions_labels, detections, size):
    """This method applies a sliding window on the data and outputs the window composed of detections and the label of this window

    Parameters
    ----------
    actions_ts : list
        All the starting action timestamps
    
    actions_durations : list
        All the durations of the actions

    actions_labels : list
        All the labels of the actions

    detections : list
        List of all the player detections

    size : int
        size in terms of time of the sliding window

    Returns
    -------
    A list of all windows and their corresponding labels
    """
    i = 0
    action_index = 0
    detection_len = len(detections)
    action_len = len(actions_ts)
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
                i = find_detection_index((start_ts + (size // 4)), detections, i) 
                yield window, "No Action"
            else:
                break
        else :
            i = find_detection_index(start_ts + (size//4), detections, i)
            yield window, actions_labels[action_index]

        if i == -1:
           break

def clean_labels(annotations):
  """This method modifies the Slow, Normal, and Fast Transitions to give them the same structure than the other actions

    Parameters
    ----------
    annotations : dict
        Dictionary containing all the informations about all the actions

    Returns
    -------
    3 lists representing the action starting timestamps, the action durations and the action labels with the Transition structure modified
    """
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

def data_to_train(window, size):
    """This method takes the window as input and outputs a list of list of size 3 with the 3 components of each person detected

    Parameters
    ----------
    window : list
        The window containing a certain amount of match detections.

    size : int
        size in terms of length of the sliding window

    Returns
    -------
    A list of list containing each 3 features per person.
    """
    size = int(size)
    data = []
    for index in range(min(len(window), size)):
        if index == size:
           break
        detection = window[index]['detections']
        player_counter = 0
        frame = np.zeros((NUM_PLAYERS + 3) * 3)
        ref_cnt = 0
        for player in range(len(detection)):
            if detection[player]['label'] == "player":
                frame[player_counter*3] = detection[player]['pos_feet'][0]
                frame[player_counter*3 + 1] = detection[player]['pos_feet'][1]
                frame[player_counter*3 + 2] = detection[player]['orientation']
                player_counter += 1
            elif ref_cnt < 3 and detection[player]['label'] == "referee":
               frame[NUM_PLAYERS*3 + ref_cnt*3] = detection[player]['pos_feet'][0]
               frame[NUM_PLAYERS*3 + ref_cnt*3 + 1] = detection[player]['pos_feet'][1]
               frame[NUM_PLAYERS*3 + ref_cnt*3 + 2] = detection[player]['orientation']
               ref_cnt += 1
            if player_counter == 10:
               break
        data.append(frame)
    if(len(window) < size):
       for _ in range(size - len(window)):
          data.append(np.zeros((NUM_PLAYERS + 3)*3))
    return np.array(data)

def sort_sequence(window):
    """This method sorts the sequence in order to add 0s when we miss a player or referee

    Parameters
    ----------
    window : list
        The window that has been outputted by data_to_train method

    Returns
    -------
    A list of list containing each 3 features per person.
    """
    sorted_window = []
    for seq in range(len(window)):
        complete_frame = np.zeros(39)
        sequence = window[seq]
        chunks = []
        for i in range(0, 30, 3):
            if(sequence[i] == 0 and sequence[i+1] == 0 and sequence[i+2] == 0):
                break
            chunks.append(sequence[i:i+3])
        chunks.sort(key=lambda x:x[0])
        chunks = np.array(chunks).flatten()
        for i in range(len(chunks)):
            complete_frame[i] = chunks[i]
        chunks = []
        for i in range(30, 39, 3):
            if(sequence[i] == 0 and sequence[i+1] == 0 and sequence[i+2] == 0):
                break
            chunks.append(sequence[i:i+3])
        chunks.sort(key=lambda x:x[0])
        chunks = np.array(chunks).flatten()
        for i in range(len(chunks)):
            complete_frame[30 + i] = chunks[i]
        sorted_window.append(complete_frame)
    return np.array(sorted_window)

def sum(n):
    #NOT USEFUL
    tot = 0
    for i in range(n):
        tot+=i
    return tot

def feature_engineering(window):
    #NOT USEFUL
    new_window = []
    for seq in range(len(window)):
        sequence = window[seq]
        new_features = np.zeros(NUM_PLAYERS*3 + 9 + sum(NUM_PLAYERS) + 2*NUM_PLAYERS + sum(3))
        new_features[:NUM_PLAYERS*3+9] = sequence
        index = 0

        #Adding the distances between the players
        for i in range(NUM_PLAYERS):
            for j in range(i+1, NUM_PLAYERS):
                distance = np.sqrt(((sequence[i*3] - sequence[j*3])**2) + ((sequence[i*3+1] - sequence[j*3+1])**2))
                new_features[NUM_PLAYERS*3+9+index] = distance
                index+= 1

        #Adding the distances between player and baskets
        for i in range(NUM_PLAYERS):
            distance_to_left = np.sqrt(((sequence[i*3] - 0)**2) + ((sequence[i*3+1] - 750))**2)
            distance_to_right = np.sqrt(((sequence[i*3] - 2800)**2) + ((sequence[i*3+1] - 750))**2)
            new_features[NUM_PLAYERS*3 + 9 + sum(NUM_PLAYERS) + 2*i] = distance_to_left
            new_features[NUM_PLAYERS*3 + 9 + sum(NUM_PLAYERS) + 2*i + 1] = distance_to_right
        
        #Adding distances between referees
        index = 0
        for i in range(3):
            for j in range(i+1, 3):
                distance = np.sqrt(((sequence[NUM_PLAYERS*3 + i*3] - sequence[NUM_PLAYERS*3 + j*3])**2) + ((sequence[NUM_PLAYERS*3 + i*3 + 1] - sequence[NUM_PLAYERS*3 + j*3 + 1])**2))
                new_features[NUM_PLAYERS*3 + 9 + sum(NUM_PLAYERS) + 2*NUM_PLAYERS + index] = distance
        new_window.append(new_features)
    return new_window
   
