import os
import time
import datetime
import bert
import data_processor
import optimization
import tokenization
import tensorflow as tf
import confusion_matrix as cf
import numpy as np
# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Disable GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Task name
task_name = 'mrda'  # TODO Needs arg?
processors = {
    "swda": data_processor.SwdaProcessor(),
    "mrda": data_processor.MrdaProcessor(),
}  # TODO add the others included with BERT

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = task_name + '_output/'
tensorboard_path = output_dir + 'tb_log'

# Training parameters
max_seq_length = 128  # TODO Needs args?
batch_size = 32
learning_rate = 2e-5
num_epochs = 2  # Default 3

print("------------------------------------")
print("Using parameters...")
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", batch_size)
print("Learning Rate: ", learning_rate)
print("Epochs: ", num_epochs)

# Configure BERT
bert_model_type = 'BERT_Base'  # TODO Needs arg?
do_lower_case = True  # TODO Needs arg?
# vocab_file = bert_model_type + '/vocab.txt'
# bert_config_file = bert_model_type + '/bert_config.json'
bert_config = bert.BertConfig.from_json_file(bert_model_type + '/bert_config.json')
if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (max_seq_length, bert_config.max_position_embeddings))

# BERT model checkpoint
init_checkpoint = bert_model_type + '/bert_model.ckpt'  # TODO Needs arg?

print("------------------------------------")
print("Configured BERT Model...")
print("Model type: ", bert_model_type)
print("Configuration: ", bert_config.to_json_string())

print("------------------------------------")
print("Preparing data...")
# Prepare data and data processors
dataset_processor = processors[task_name.lower()]
tokenizer = tokenization.FullTokenizer(vocab_file=bert_model_type + '/vocab.txt', do_lower_case=do_lower_case)

# Get labels
labels = dataset_processor.get_labels(data_dir)

# train_data_dir = output_dir + 'train/'
train_data_file = os.path.join(output_dir, "train.tf_record")
training_examples = dataset_processor.get_train_examples(data_dir)
num_training_steps = int(len(training_examples) / batch_size)
data_processor.convert_examples_to_features(training_examples, labels, max_seq_length, tokenizer, train_data_file)

# eval_data_dir = output_dir + 'eval'
eval_data_file = os.path.join(output_dir, "eval.tf_record")
evaluation_examples = dataset_processor.get_eval_examples(data_dir)
num_evaluation_steps = int(len(evaluation_examples) / batch_size)
data_processor.convert_examples_to_features(evaluation_examples, labels, max_seq_length, tokenizer, eval_data_file)

train_dataset = data_processor.build_dataset(train_data_file, max_seq_length, batch_size, is_training=True)
eval_dataset = data_processor.build_dataset(eval_data_file, max_seq_length, batch_size, is_training=False)

handle_pl = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dataset.output_types, train_dataset.output_shapes)
# iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
#                                            train_dataset.output_shapes)
iterator_next = iterator.get_next()

input_ids = tf.convert_to_tensor(iterator_next["input_ids"], name='input_ids')
input_mask = tf.convert_to_tensor(iterator_next["input_mask"], name='input_mask')
segment_ids = tf.convert_to_tensor(iterator_next["segment_ids"], name='segment_ids')
label_ids = tf.convert_to_tensor(iterator_next["label_ids"], name='label_ids')

train_iterator = train_dataset.make_initializable_iterator()
eval_iterator = eval_dataset.make_initializable_iterator()
# train_iterator = iterator.make_initializer(train_dataset)
# eval_iterator = iterator.make_initializer(eval_dataset)

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

# tvars = tf.trainable_variables()
# initialized_variable_names = {}
# if init_checkpoint:
# assignment_map, initialized_variable_names = bert.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
# tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


saver = tf.train.Saver()
# Run Tensorflow session
with tf.Session() as sess:
    # saver = tf.train.import_meta_graph('BERT_Base/bert_model.ckpt.meta') # TODO possible save/restore alternative
    # saver.restore(sess, 'mrda_output/test.ckpt')
    print("Model restored.")

    # Create Tensorboard writers for the training and test data
    train_writer = tf.summary.FileWriter('%s/%s' % (tensorboard_path, 'train'), sess.graph)
    eval_writer = tf.summary.FileWriter('%s/%s' % (tensorboard_path, 'eval'), sess.graph)
    summary = tf.summary.merge([accuracy_summary, loss_summary])

    # Initialise all the variables
    sess.run(tf.global_variables_initializer())

    # Train the model
    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for", num_epochs, "epochs")

    num_global_steps = num_training_steps * num_epochs
    global_step = 0
    # Run for number of training epochs
    for epoch in range(1, num_epochs + 1):

        # Initialise the iterator handles
        # sess.run(iterator.make_initializer(train_dataset))
        train_handle = sess.run(train_iterator.string_handle())
        eval_handle = sess.run(eval_iterator.string_handle())

        # For accumulating the epoch loss and accuracy stats
        train_loss = []
        train_accuracy = []
        eval_loss = []
        eval_accuracy = []

        # Initialise the training dataset
        sess.run(train_iterator.initializer)
        for train_step in range(1, num_training_steps + 1):
            # Need to switch model to not training
            model.is_training = True
            global_step += 1

            _, train_batch_loss, train_batch_logits, train_batch_accuracy = \
                sess.run([optimizer, loss, logits, accuracy], feed_dict={handle_pl: train_handle})
            train_loss.append(train_batch_loss)
            train_accuracy.append(train_batch_accuracy)

            # Evaluate every X training steps
            if train_step % 5 == 0:

                # For accumulating this evaluations loss and accuracy stats
                eval_step_loss = []
                eval_step_accuracy = []
                # For accumulating this evaluations predictions for confusion matrix
                eval_step_predictions = []
                eval_step_labels = []

                # Initialise the evaluation dataset
                sess.run(eval_iterator.initializer)
                for eval_step in range(num_evaluation_steps):
                    # Need to switch model to not training
                    model.is_training = False

                    _, eval_batch_loss, eval_batch_logits, eval_batch_accuracy, eval_batch_predictions, eval_batch_labels = \
                        sess.run([optimizer, loss, logits, accuracy, predictions, label_ids], feed_dict={handle_pl: eval_handle})
                    eval_loss.append(eval_batch_loss)
                    eval_accuracy.append(eval_batch_accuracy)
                    eval_step_loss.append(eval_batch_loss)
                    eval_step_accuracy.append(eval_batch_accuracy)
                    eval_step_predictions = np.concatenate((eval_step_predictions, eval_batch_predictions), axis=None)
                    eval_step_labels = np.concatenate((eval_step_labels, eval_batch_labels), axis=None)

                # Record training summaries
                train_summary = sess.run(summary, feed_dict={accuracy_pl: sum(train_accuracy) / len(train_accuracy),
                                                             loss_pl: sum(train_loss)/len(train_loss)})
                eval_summary = sess.run(summary, feed_dict={accuracy_pl: sum(eval_step_accuracy) / len(eval_step_accuracy),
                                                            loss_pl: sum(eval_step_loss) / len(eval_step_loss)})
                train_writer.add_summary(train_summary, global_step)
                eval_writer.add_summary(eval_summary, global_step)

                # Record confusion matrix
                cm_summary, cm_fig = cf.plot_confusion_matrix(eval_step_labels, eval_step_predictions, labels, tensor_name='eval_confusion_matrix')
                eval_writer.add_summary(cm_summary, global_step)

                # Display epoch statistics
                print("Step: {}/{} - "
                      "Training loss: {:.3f}, acc: {:.2f}% - "
                      "Evaluation loss: {:.3f}, acc: {:.2f}%".format(global_step, num_global_steps,
                                                                     (sum(train_loss) / len(train_loss)),
                                                                     ((sum(train_accuracy) * 100) / len(train_accuracy)),
                                                                     (sum(eval_step_loss) / len(eval_step_loss)),
                                                                     ((sum(eval_step_accuracy) * 100) / len(eval_step_accuracy))))
        # Display epoch statistics
        print("Epoch: {}/{} - "
              "Training loss: {:.3f}, acc: {:.2f}% - "
              "Evaluation loss: {:.3f}, acc: {:.2f}%".format(epoch, num_epochs,
                                                             (sum(train_loss) / len(train_loss)),
                                                             ((sum(train_accuracy) * 100) / len(train_accuracy)),
                                                             (sum(eval_loss) / len(eval_loss)),
                                                             ((sum(eval_accuracy) * 100) / len(eval_accuracy))))

        # saver.save(sess, 'mrda_output/test.ckpt')
    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for", num_epochs, "epochs")
