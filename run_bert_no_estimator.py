import os
import sys
import logger
import time
import datetime
import bert
import data_processor
import optimization
import tokenization
import tensorflow as tf
import confusion_matrix as cf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Disable GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set number of CPU Cores to use
config = tf.ConfigProto(device_count={"CPU": 24},
                        inter_op_parallelism_threads=24,
                        intra_op_parallelism_threads=24)
tf.Session(config=config)

# Task name
task_name = 'swda'  # TODO Needs arg?
processors = {
    "swda": data_processor.SwdaProcessor(),
    "mrda": data_processor.MrdaProcessor(),
}  # TODO add the others included with BERT

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = task_name + '_output/'
tensorboard_dir = output_dir + 'tb_log'
model_dir = output_dir + 'model/'
# Save std out to log file
log_file = output_dir + "output.txt"
sys.stdout = logger.Logger(log_file)

# Training parameters
max_seq_length = 128  # TODO Needs args?
batch_size = 32
learning_rate = 2e-5
num_epochs = 3  # Default 3
evaluation_steps = 500  # Number of evaluations to make per epoch
training = True
testing = True

print("------------------------------------")
print("Using parameters...")
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", batch_size)
print("Learning rate: ", learning_rate)
print("Epochs: ", num_epochs)
print("Evaluation steps: ", evaluation_steps)
print("Training: ", training)
print("Testing: ", testing)

# Configure BERT
bert_model_type = 'BERT_Base'  # TODO Needs arg?
do_lower_case = True  # TODO Needs arg?
bert_config = bert.BertConfig.from_json_file(bert_model_type + '/bert_config.json')
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
train_data_file = os.path.join(output_dir, "train.tf_record")
training_examples = dataset_processor.get_train_examples(data_dir)
data_processor.convert_examples_to_features(training_examples, labels, max_seq_length, tokenizer, train_data_file)
train_dataset = data_processor.build_dataset(train_data_file, max_seq_length, batch_size, is_training=True)

# Evaluation data
eval_data_file = os.path.join(output_dir, "eval.tf_record")
evaluation_examples = dataset_processor.get_eval_examples(data_dir)
data_processor.convert_examples_to_features(evaluation_examples, labels, max_seq_length, tokenizer, eval_data_file)
eval_dataset = data_processor.build_dataset(eval_data_file, max_seq_length, batch_size, is_training=False)

# Test data
test_data_file = os.path.join(output_dir, "test.tf_record")
test_examples = dataset_processor.get_test_examples(data_dir)
data_processor.convert_examples_to_features(test_examples, labels, max_seq_length, tokenizer, test_data_file)
test_dataset = data_processor.build_dataset(test_data_file, max_seq_length, batch_size, is_training=False)

# Create iterators
handle_pl = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dataset.output_types, train_dataset.output_shapes)
iterator_next = iterator.get_next()

input_ids = tf.convert_to_tensor(iterator_next["input_ids"], name='input_ids')
input_mask = tf.convert_to_tensor(iterator_next["input_mask"], name='input_mask')
segment_ids = tf.convert_to_tensor(iterator_next["segment_ids"], name='segment_ids')
label_ids = tf.convert_to_tensor(iterator_next["label_ids"], name='label_ids')

train_iterator = train_dataset.make_initializable_iterator()
eval_iterator = eval_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()

# Set number of training, evaluation and test steps (+1 so we dont miss the last partial batch)
num_training_steps = int(len(training_examples) / batch_size) + 1
num_evaluation_steps = int(len(evaluation_examples) / batch_size) + 1
num_test_steps = int(len(test_examples) / batch_size) + 1

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
model = bert.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

# In the demo, we are doing a simple classification task on the entire segment.
# If you want to use the token-level output, use model.get_sequence_output() instead.
output_layer = model.get_pooled_output()

hidden_size = output_layer.shape[-1].value

output_weights = tf.get_variable(
    "output_weights", [len(labels), hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

output_bias = tf.get_variable(
    "output_bias", [len(labels)], initializer=tf.zeros_initializer())

# Calculate the cost
with tf.variable_scope("loss"):
    # if is_training:
    if model.is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(label_ids, depth=len(labels), dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    loss_pl = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('Loss', loss_pl)

# Optimisation function
with tf.name_scope('optimizer'):
    optimizer = optimization.create_optimizer(loss, learning_rate, num_training_steps, 0, False)

# Calculate accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(predictions, tf.cast(label_ids, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_pl = tf.placeholder(tf.float32)
    accuracy_summary = tf.summary.scalar('Accuracy', accuracy_pl)

# Restore model from checkpoint
trainable_vars = tf.trainable_variables()
if init_checkpoint:
    assignment_map, _ = bert.get_assignment_map_from_checkpoint(trainable_vars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

saver = tf.train.Saver()
print("Model restored from " + init_checkpoint)

# Run Tensorflow session
with tf.Session() as sess:
    # Remove old Tensorboard directory
    if training:
        if tf.gfile.Exists(tensorboard_dir + '/train'):
            tf.gfile.DeleteRecursively(tensorboard_dir + '/train')
        if tf.gfile.Exists(tensorboard_dir + '/eval'):
            tf.gfile.DeleteRecursively(tensorboard_dir + '/eval')
    if testing:
        if tf.gfile.Exists(tensorboard_dir + '/test'):
            tf.gfile.DeleteRecursively(tensorboard_dir + '/test')

    # Create Tensorboard writers for the training and test data
    train_writer = tf.summary.FileWriter('%s/%s' % (tensorboard_dir, 'train'), sess.graph)
    eval_writer = tf.summary.FileWriter('%s/%s' % (tensorboard_dir, 'eval'), sess.graph)
    test_writer = tf.summary.FileWriter('%s/%s' % (tensorboard_dir, 'test'), sess.graph)
    summary = tf.summary.merge([accuracy_summary, loss_summary])

    # Initialise all the variables
    sess.run(tf.global_variables_initializer())

    # Initialise the iterator handles
    train_handle = sess.run(train_iterator.string_handle())
    eval_handle = sess.run(eval_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    # Total number of training steps
    num_global_steps = num_training_steps * num_epochs

    if training:
        # Train the model
        print("------------------------------------")
        print("Training model...")
        start_time = time.time()
        print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for", num_epochs, "epochs")

        global_step = 0
        # Run for number of training epochs
        for epoch in range(1, num_epochs + 1):

            # For accumulating the epoch loss and accuracy stats
            train_loss = []
            train_accuracy = []
            eval_loss = []
            eval_accuracy = []

            # Initialise the training dataset
            sess.run(train_iterator.initializer)
            for train_step in range(1, num_training_steps + 1):
                # Need to switch model to training
                model.is_training = True
                global_step += 1

                _, train_batch_loss, train_batch_logits, train_batch_accuracy = \
                    sess.run([optimizer, loss, logits, accuracy], feed_dict={handle_pl: train_handle})
                train_loss.append(train_batch_loss)
                train_accuracy.append(train_batch_accuracy)

                # Evaluate every evaluation_steps during training
                if train_step % evaluation_steps == 0:

                    # For accumulating this evaluations loss and accuracy stats
                    eval_step_loss = []
                    eval_step_accuracy = []
                    # For accumulating this evaluations predictions for confusion matrix
                    eval_step_predictions = []
                    eval_step_labels = []

                    # Initialise the evaluation dataset
                    sess.run(eval_iterator.initializer)
                    for eval_step in range(1, num_evaluation_steps + 1):
                        # Need to switch model to not training
                        model.is_training = False

                        eval_batch_loss, eval_batch_logits, eval_batch_accuracy, eval_batch_predictions, eval_batch_labels = \
                            sess.run([loss, logits, accuracy, predictions, label_ids],
                                     feed_dict={handle_pl: eval_handle})
                        eval_loss.append(eval_batch_loss)
                        eval_accuracy.append(eval_batch_accuracy)
                        eval_step_loss.append(eval_batch_loss)
                        eval_step_accuracy.append(eval_batch_accuracy)
                        eval_step_predictions = np.concatenate((eval_step_predictions, eval_batch_predictions),
                                                               axis=None)
                        eval_step_labels = np.concatenate((eval_step_labels, eval_batch_labels), axis=None)

                    # Record training summaries
                    train_summary = sess.run(summary,
                                             feed_dict={accuracy_pl: sum(train_accuracy) / len(train_accuracy),
                                                        loss_pl: sum(train_loss) / len(train_loss)})
                    train_writer.add_summary(train_summary, global_step)

                    eval_summary = sess.run(summary,
                                            feed_dict={accuracy_pl: sum(eval_step_accuracy) / len(eval_step_accuracy),
                                                       loss_pl: sum(eval_step_loss) / len(eval_step_loss)})
                    eval_writer.add_summary(eval_summary, global_step)

                    # Record confusion matrix
                    cm_summary, cm_fig = cf.plot_confusion_matrix(eval_step_labels, eval_step_predictions, labels,
                                                                  tensor_name='eval_confusion_matrix')
                    eval_writer.add_summary(cm_summary, global_step)

                    # Display step statistics
                    print("Step: {}/{} - "
                          "Training loss: {:.3f}, accuracy: {:.3f}% - "
                          "Evaluation loss: {:.3f}, accuracy: {:.3f}%".format(global_step, num_global_steps,
                                                                         (sum(train_loss) / len(train_loss)),
                                                                         ((sum(train_accuracy) * 100) / len(
                                                                             train_accuracy)),
                                                                         (sum(eval_step_loss) / len(eval_step_loss)),
                                                                         ((sum(eval_step_accuracy) * 100) / len(
                                                                             eval_step_accuracy))))

                    # Save the model
                    saver.save(sess, model_dir + task_name, global_step=global_step, write_meta_graph=False)

            # Display epoch statistics
            print("Epoch: {}/{} - "
                  "Training loss: {:.3f}, accuracy: {:.3f}% - "
                  "Evaluation loss: {:.3f}, accuracy: {:.3f}%".format(epoch, num_epochs,
                                                                 (sum(train_loss) / len(train_loss)),
                                                                 ((sum(train_accuracy) * 100) / len(train_accuracy)),
                                                                 (sum(eval_loss) / len(eval_loss)),
                                                                 ((sum(eval_accuracy) * 100) / len(eval_accuracy))))

            # If this is the last global step then save the model
            if global_step == num_global_steps:
                saver.save(sess, model_dir + task_name, global_step=global_step, write_meta_graph=False)

        end_time = time.time()
        print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for", num_epochs, "epochs")

    if testing:
        # Test the model
        print("------------------------------------")
        print("Testing model...")
        start_time = time.time()
        print("Testing started: " + datetime.datetime.now().strftime("%b %d %T"))

        # For accumulating the test loss and accuracy stats
        test_loss = []
        test_accuracy = []
        # For accumulating this evaluations predictions for confusion matrix
        test_probabilities = []
        test_predictions = []
        test_labels = []

        # Need to switch model to not training
        model.is_training = False

        # Initialise the training dataset
        sess.run(test_iterator.initializer)
        for test_step in range(1, num_test_steps + 1):
            test_batch_loss, test_batch_logits, test_batch_accuracy, test_batch_probabilities, test_batch_predictions, test_batch_labels = \
                sess.run([loss, logits, accuracy, probabilities, predictions, label_ids],
                         feed_dict={handle_pl: test_handle})
            test_loss.append(test_batch_loss)
            test_accuracy.append(test_batch_accuracy)
            for i in range(len(test_batch_probabilities)):
                test_probabilities.append(test_batch_probabilities[i])
            test_predictions = np.concatenate((test_predictions, test_batch_predictions), axis=None)
            test_labels = np.concatenate((test_labels, test_batch_labels), axis=None)

        # Record training summaries
        test_summary = sess.run(summary, feed_dict={accuracy_pl: sum(test_accuracy) / len(test_accuracy),
                                                    loss_pl: sum(test_loss) / len(test_loss)})
        test_writer.add_summary(test_summary, num_global_steps)

        # Record confusion matrix
        cm_summary, cm_fig = cf.plot_confusion_matrix(test_labels, test_predictions, labels,
                                                      tensor_name='test_confusion_matrix')
        test_writer.add_summary(cm_summary, num_global_steps)

        # Write the prediction results to a file
        test_predictions_file = os.path.join(output_dir, "test_predictions.csv")
        with open(test_predictions_file, "w") as file:
            for prediction in test_probabilities:
                output_line = ",".join(str(class_probability) for class_probability in prediction) + "\n"
                file.write(output_line)

        test_metrics = dict()
        # Calculate precision, recall and F1
        with tf.variable_scope("F1_macro"):
            precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(test_labels, test_predictions, average='macro')
            test_metrics['Precision_macro'] = precision_mac
            test_metrics['Recall_macro'] = recall_mac
            test_metrics['F1_macro'] = precision_mac
            precision_mac_summary = tf.summary.scalar('Precision_macro', precision_mac)
            recall_mac_summary = tf.summary.scalar('Recall_macro', recall_mac)
            f1_mac_summary = tf.summary.scalar('F1_macro', f1_mac)

        with tf.variable_scope("F1_micro"):
            precision_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(test_labels, test_predictions, average='micro')
            test_metrics['Precision_micro'] = precision_mic
            test_metrics['Recall_micro'] = recall_mic
            test_metrics['F1_micro'] = f1_mic
            precision_mic_summary = tf.summary.scalar('Precision_micro', precision_mic)
            recall_mic_summary = tf.summary.scalar('Recall_micro', recall_mic)
            f1_mic_summary = tf.summary.scalar('F1_micro', f1_mic)

        with tf.variable_scope("F1_weighted"):
            precision_weight, recall_weight, f1_weight, _ = precision_recall_fscore_support(test_labels, test_predictions, average='weighted')
            test_metrics['Precision_weighted'] = precision_weight
            test_metrics['Recall_weighted'] = recall_weight
            test_metrics['F1_weighted'] = f1_weight
            precision_weight_summary = tf.summary.scalar('Precision_weighted', precision_weight)
            recall_weight_summary = tf.summary.scalar('Recall_weighted', recall_weight)
            f1_weight_summary = tf.summary.scalar('F1_weighted', f1_weight)

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

        # Display test statistics
        print("Testing loss: {:.3f}, accuracy: {:.3f}%".format((sum(test_loss) / len(test_loss)),
                                                          (sum(test_accuracy) * 100) / len(test_accuracy)))

        for key in sorted(test_metrics.keys()):
            print("{}: {:.3f}".format(key, test_metrics[key]))

        end_time = time.time()
        print("Testing took " + str(('%.3f' % (end_time - start_time))) + " seconds")
