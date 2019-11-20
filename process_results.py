import os
import csv
import data_processor
import numpy as np
import pickle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#import confusion_matrix as cm

# Task name
task_name = 'swda'  # TODO Needs arg?
processors = {
    "swda": data_processor.SwdaProcessor(),
    "mrda": data_processor.MrdaProcessor(),
}  # TODO add the others included with BERT

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = task_name + '_output'

test_results_file = os.path.join(output_dir, 'test', "test_predictions.csv")
test_results = []
test_metrics = dict()

with open(test_results_file) as file:
    file_data = csv.reader(file, delimiter=',')

    test_predictions = list(file_data)
    test_predictions = np.array(test_predictions).astype(float)


# Get the actual labels from the data and convert to one hot
processor = processors[task_name.lower()]
test_examples = processor.get_test_examples(data_dir)
labels = processor.get_labels(data_dir)

# Get the actual labels from the data and convert to one hot
true_labels = [test_examples[i].label for i in range(len(test_examples))]
true_labels = label_binarize(true_labels, labels)

# Get the index of the prediction/actual labels
true_labels = [np.argmax(label) for label in true_labels]
predicted_labels = [np.argmax(prediction) for prediction in test_predictions]

# Calculate the accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
test_metrics['Accuracy'] = accuracy
print(accuracy)

# test_results_file = os.path.join(output_dir, "test", "test_results.txt")
# with open(test_results_file, "w+") as file:
#     print("***** Test results *****")
#     file.write("***** Test results *****\n")
#     for key in sorted(test_metrics.keys()):
#         print("%s = %s" % (key, str(test_metrics[key])))
#         file.write("%s = %s\n" % (key, str(test_metrics[key])))
