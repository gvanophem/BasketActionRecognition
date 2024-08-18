from sklearn.metrics import precision_score, recall_score, classification_report

#import torch
from collections import Counter
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import top_k_accuracy_score
import seaborn as sns

thirdDense = [32,64,128, 256, 512, 1024]
accuracies = [22, 27, 31,  39, 42, 40]

plt.plot(thirdDense, accuracies)
plt.xlabel('Size of Transformed CLS')
plt.ylabel('Accuracy')
plt.title('Accuracy of the Model Based on the Size of the Transformed CLS')
plt.xscale('log')
plt.savefig("sizeCLS.png")
plt.show()


def data_for_iou(y):
    detections = []
    i=0
    while i<len(y):
        label = y[i]
        j = 0
        while i+j < len(y) and y[i+j] == label:
            j+=1
        detections.append([label, i, i+j])
        i+=j
    return detections

def intersection_on_union(detection, truth):
    first_interval = detection[1:]
    second_interval = truth[1:]

    intersection = max(0, min(first_interval[1],second_interval[1]) - max(first_interval[0],second_interval[0]))
    union = (first_interval[1]-first_interval[0]) + (second_interval[1]-second_interval[0]) - intersection

    if union==0 : 
        return 0

    return intersection/union
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=11):

    # code inspired by https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py
    # found on a youtube video of Aladdin Persson

    average_precisions = []

    for c in range(num_classes):

        detections = []
        ground_truths=[]

        for detection in pred_boxes:
            if detection[0] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[0] == c:
                ground_truths.append(true_box)

        amount_bboxes = np.zeros(len(ground_truths))

        TP = np.zeros((len(detections)))
        FP = np.zeros((len(detections)))
        total_true_boxes = len(ground_truths)

        if total_true_boxes == 0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            best_iou = 0

            for idx, gt in enumerate(ground_truths):
                iou = intersection_on_union(detection, gt)

                if iou > best_iou :
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[best_gt_idx] == 0:
                    amount_bboxes[best_gt_idx] = 1
                    TP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        recalls = TP_cumsum/total_true_boxes
        precisions = TP_cumsum/(TP_cumsum+FP_cumsum)
        precisions = np.concatenate(([1], precisions))
        recalls = np.concatenate(([0], recalls))

        average_precisions.append(np.trapz(precisions, x=recalls))
    return average_precisions

with open("data_size400_epochs400_embed512_thirdDense512.npy", 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    c = np.load(f)

print(np.sum(a==b)/len(a))


cm = confusion_matrix(a,b)

"""disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()"""

# Plot confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set plot labels
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for LSTM')
plt.savefig('confusion.png')
plt.show()

top_accuracies = []
x=[]
for i in range(1, 12):
    top_accuracies.append(top_k_accuracy_score(a,c, k=i))
    x.append(i)

print(top_accuracies)

plt.plot(x,top_accuracies)
plt.xticks(x)
plt.xlabel('Value of k')
plt.ylabel('Accuracy')
plt.title('Accuracy of LSTM based on the value of k')
plt.savefig('top_k.png')
plt.show()


print(classification_report(a,b))


a_boxes = data_for_iou(a)
b_boxes = data_for_iou(b)

print(mean_average_precision(b_boxes, a_boxes, iou_threshold=0.05))


labels = ['No Game', 'Half-court', 'Slow Transition', 'EOB', 'SOB', 'Free Throw', 'Timeout', 'Jump Ball', 'Normal Transition', 'Fast Transition', 'No Action']
maps=[]
k = 0
x = []
for i in range (40):
    k += 0.025
    x.append(k)
    AP = mean_average_precision(b_boxes, a_boxes, iou_threshold=k)
    maps.append(AP)

for i in range(11):
    y = []
    for j in range(40):
        y.append(maps[j][i])
    label = labels[i]
    if(label == 'No Game'):
        plt.plot(x,y,label=label, color='black')
    else:   
        plt.plot(x,y, label=label)

plt.xlabel('Value of k')
plt.ylabel('Average Precision')
plt.ylim((0, 0.3))
plt.title('Evolution of the Average Precision with k')
plt.legend()
plt.savefig("mAP.png")
plt.show()