import bert
import data_processor
import tokenization
import numpy as np
import bert_utilities as utils
from best_checkpoint_copier import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import confusion_matrix as cm

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)
# Disable GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set number of CPU Cores to use
config = tf.ConfigProto(device_count={"CPU": 24},
                        inter_op_parallelism_threads=24,
                        intra_op_parallelism_threads=24)
tf.Session(config=config)

# Task name
task_name = 'mrda'  # TODO Needs args?
experiment_name = 'full_tags_15_epoch'
processors = {
    "cola": data_processor.ColaProcessor,
    "mnli": data_processor.MnliProcessor,
    "mrpc": data_processor.MrpcProcessor,
    "xnli": data_processor.XnliProcessor,
    "swda": data_processor.SwdaProcessor(),
    "mrda": data_processor.MrdaProcessor(),
}

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = task_name + '_output'

# Create appropriate directories if they don't exist
if experiment_name is not '' and experiment_name is not None:
    output_dir = os.path.join(output_dir, experiment_name)

datasets_dir = os.path.join(output_dir, 'datasets')
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

# Training parameters
max_seq_length = 128  # Default 128 TODO Needs args?
batch_size = 32  # Default 32
learning_rate = 2e-5  # Default 2e-5
num_epochs = 15.0  # Default 3
save_checkpoint_steps = 1000  # 1000
evaluate_secs = 3600  # 3600
checkpoints_to_keep = 1
training = True
testing = True

print("------------------------------------")
print("Using parameters...")
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", batch_size)
print("Learning Rate: ", learning_rate)
print("Epochs: ", num_epochs)
print("Save checkpoints every " + str(save_checkpoint_steps) + " steps")
print("Evaluate every " + str(evaluate_secs) + " seconds")
print("Training: ", training)
print("Testing: ", testing)

# Configure BERT
bert_model_type = 'BERT_Base'  # TODO Needs arg?
do_lower_case = True  # TODO Needs arg?
bert_config = bert.BertConfig.from_json_file(bert_model_type + '/bert_config.json')
if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (max_seq_length, bert_config.max_position_embeddings))

# BERT model checkpoint
init_checkpoint = bert_model_type + '/bert_model.ckpt'

print("------------------------------------")
print("Configured BERT Model...")
print("Model type: ", bert_model_type)
print("Configuration: ", bert_config.to_json_string())

# Prepare data and data processors
dataset_processor = processors[task_name.lower()]
tokenizer = tokenization.FullTokenizer(vocab_file=bert_model_type + '/vocab.txt', do_lower_case=do_lower_case)

# Get labels
labels = dataset_processor.get_labels(data_dir)

# Training data
training_examples = dataset_processor.get_train_examples(data_dir)
train_data_file = os.path.join(datasets_dir, "train.tf_record")
data_processor.convert_examples_to_features(training_examples, labels, max_seq_length, tokenizer, train_data_file)
train_input_fn = utils.input_fn_builder(
    train_data_file,
    max_seq_length,
    is_training=True,
    drop_remainder=True)

# Evaluation data
evaluation_examples = dataset_processor.get_eval_examples(data_dir)
eval_data_file = os.path.join(datasets_dir, "eval.tf_record")
data_processor.convert_examples_to_features(evaluation_examples, labels, max_seq_length, tokenizer, eval_data_file)
eval_input_fn = utils.input_fn_builder(
    input_file=eval_data_file,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=False)

# Test data
test_examples = dataset_processor.get_test_examples(data_dir)
test_data_file = os.path.join(datasets_dir, "test.tf_record")
data_processor.convert_examples_to_features(test_examples, labels, max_seq_length, tokenizer, test_data_file)
test_input_fn = utils.input_fn_builder(
    input_file=test_data_file,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=False)

# Set number of training, evaluation and test steps
num_training_steps = int(len(training_examples) / batch_size * num_epochs)
num_evaluation_steps = int(len(evaluation_examples) / batch_size)
num_test_steps = int(len(test_examples) / batch_size)

# Copies the best checkpoint to far to its own directory
best_checkpoint_copier = BestCheckpointCopier(
   name='best_checkpoint',
   checkpoints_to_keep=checkpoints_to_keep,
   score_metric='loss',
   compare_fn=lambda x, y: x.score < y.score,
   sort_key_fn=lambda x: x.score,
   sort_reverse=False)

print("------------------------------------")
print("Prepared data...")
print("Number of labels: ", len(labels))
print("Number of training examples: ", len(training_examples))
print("Number of training steps: ", num_training_steps)
print("Number of evaluation examples: ", len(evaluation_examples))
print("Number of evaluation steps: ", num_evaluation_steps)
print("Number of test examples: ", len(test_examples))
print("Number of test steps: ", num_test_steps)

# Define BERT Model
print("------------------------------------")
print("Define BERT Model...")
model_fn = utils.model_fn_builder(
    bert_config=bert_config,
    num_labels=len(labels),
    init_checkpoint=init_checkpoint,
    learning_rate=learning_rate,
    num_train_steps=num_training_steps,
    num_warmup_steps=0,
    use_tpu=False,
    use_one_hot_embeddings=False)

run_config = tf.estimator.RunConfig(model_dir=output_dir, save_checkpoints_steps=save_checkpoint_steps, save_summary_steps=save_checkpoint_steps)
estimator = tf.estimator.Estimator(model_fn, model_dir=output_dir, config=run_config, params={'batch_size': batch_size})

if training:
    # Train BERT
    print("------------------------------------")
    print("Train Model...")
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_training_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=best_checkpoint_copier, steps=num_evaluation_steps, throttle_secs=evaluate_secs)
    eval_metrics = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write the evaluation results to a file
    eval_results_file = os.path.join(output_dir, "eval", "eval_results.txt")
    with open(eval_results_file, "w") as file:
        print("***** Evaluation results *****")
        file.write("***** Evaluation results *****\n")
        for key in eval_metrics[0].keys():
            print("%s = %s" % (key, str(eval_metrics[0][key])))
            file.write("%s = %s\n" % (key, str(eval_metrics[0][key])))

if testing:
    # Test BERT
    print("------------------------------------")
    print("Test Model...")
    test_results = estimator.predict(input_fn=test_input_fn)

    # Make a copy because otherwise TF likes to delete the results after accessing
    test_predictions = [prediction for prediction in test_results]
    test_metrics = dict()

    # Get the actual labels from the data and convert to one hot
    true_labels = [test_examples[i].label for i in range(len(test_examples))]
    true_labels = label_binarize(true_labels, labels)

    # Get the index of the prediction/actual labels
    true_labels = [np.argmax(label) for label in true_labels]
    predicted_labels = [np.argmax(prediction) for prediction in test_predictions]

    # Calculate the accuracy
    with tf.variable_scope("accuracy"):
        accuracy = accuracy_score(true_labels, predicted_labels)
        test_metrics['Accuracy'] = accuracy
        accuracy_summary = tf.summary.scalar('Test_accuracy', accuracy)

    # Calculate precision, recall and F1
    with tf.variable_scope("F1_macro"):
        precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
        test_metrics['Precision_macro'] = precision_mac
        test_metrics['Recall_macro'] = recall_mac
        test_metrics['F1_macro'] = precision_mac
        precision_mac_summary = tf.summary.scalar('Precision_macro', precision_mac)
        recall_mac_summary = tf.summary.scalar('Recall_macro', recall_mac)
        f1_mac_summary = tf.summary.scalar('F1_macro', f1_mac)

    with tf.variable_scope("F1_micro"):
        precision_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')
        test_metrics['Precision_micro'] = precision_mic
        test_metrics['Recall_micro'] = recall_mic
        test_metrics['F1_micro'] = f1_mic
        precision_mic_summary = tf.summary.scalar('Precision_micro', precision_mic)
        recall_mic_summary = tf.summary.scalar('Recall_micro', recall_mic)
        f1_mic_summary = tf.summary.scalar('F1_micro', f1_mic)

    with tf.variable_scope("F1_weighted"):
        precision_weight, recall_weight, f1_weight, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
        test_metrics['Precision_weighted'] = precision_weight
        test_metrics['Recall_weighted'] = recall_weight
        test_metrics['F1_weighted'] = f1_weight
        precision_weight_summary = tf.summary.scalar('Precision_weighted', precision_weight)
        recall_weight_summary = tf.summary.scalar('Recall_weighted', recall_weight)
        f1_weight_summary = tf.summary.scalar('F1_weighted', f1_weight)

    # Add the summaries to tensorboard
    with tf.Session() as sess:
        test_writer = tf.summary.FileWriter('%s/%s' % (output_dir, 'test'), sess.graph)

        # Accuracy
        test_writer.add_summary(accuracy_summary.eval(), num_training_steps)

        # Macro F1
        test_writer.add_summary(precision_mac_summary.eval(), num_training_steps)
        test_writer.add_summary(recall_mac_summary.eval(), num_training_steps)
        test_writer.add_summary(f1_mac_summary.eval(), num_training_steps)

        # Micro F1
        test_writer.add_summary(precision_mic_summary.eval(), num_training_steps)
        test_writer.add_summary(recall_mic_summary.eval(), num_training_steps)
        test_writer.add_summary(f1_mic_summary.eval(), num_training_steps)

        # Weighted F1
        test_writer.add_summary(precision_weight_summary.eval(), num_training_steps)
        test_writer.add_summary(recall_weight_summary.eval(), num_training_steps)
        test_writer.add_summary(f1_weight_summary.eval(), num_training_steps)

        # Confusion Matrix
        cm_summary, matrix = cm.plot_confusion_matrix(true_labels, predicted_labels, labels)
        test_writer.add_summary(cm_summary, num_training_steps)

    # Write the results to a file
    test_predictions_file = os.path.join(output_dir, "test", "test_predictions.csv")
    with open(test_predictions_file, "w") as file:
        for prediction in test_predictions:
            output_line = ",".join(str(class_probability) for class_probability in prediction) + "\n"
            file.write(output_line)

    test_results_file = os.path.join(output_dir, "test", "test_results.txt")
    with open(test_results_file, "w+") as file:
        print("***** Test results *****")
        file.write("***** Test results *****\n")
        for key in sorted(test_metrics.keys()):
            print("%s = %s" % (key, str(test_metrics[key])))
            file.write("%s = %s\n" % (key, str(test_metrics[key])))
