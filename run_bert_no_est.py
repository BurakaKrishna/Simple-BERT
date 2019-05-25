import collections
import csv
import os
import pickle

# Disable GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import bert
import bert_utilities as utils
import data_processor
import optimization
import tokenization
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Task name
task_name = 'mrda'  # TODO Needs arg?
processors = {
    "swda": data_processor.SwdaProcessor(),
    "mrda": data_processor.MrdaProcessor(),
}  # TODO add the others included with BERT

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = task_name + '_output/'

# Training parameters
max_seq_length = 128  # TODO Needs args?
train_batch_size = 32
learning_rate = 2e-5
num_train_epochs = 2  # Default 3

print("------------------------------------")
print("Using parameters...")
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", train_batch_size)
print("Learning Rate: ", learning_rate)
print("Epochs: ", num_train_epochs)

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
label_list = dataset_processor.get_labels(data_dir)

train_data_dir = output_dir + "/train"
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
# Get training data and input function
training_examples = dataset_processor.get_train_examples(data_dir)
num_training_steps = int(len(training_examples) / train_batch_size * num_train_epochs)
train_data_file = os.path.join(train_data_dir, "train.tf_record")
data_processor.convert_examples_to_features(training_examples, label_list, max_seq_length, tokenizer, train_data_file)
dataset = tf.data.TFRecordDataset(train_data_file).shuffle(buffer_size=100).repeat(1)
print(dataset)
name_to_features = {
    "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
    "label_ids": tf.FixedLenFeature([], tf.int64),
}
dataset = dataset.apply(tf.contrib.data.map_and_batch(lambda record: utils.decode_record(record, name_to_features),
                                                      batch_size=train_batch_size,
                                                      drop_remainder=False))
iterator = dataset.make_initializable_iterator()
# iterator = tf.data.Iterator.from_structure(dataset.output_types,
#                                            dataset.output_shapes)
next = iterator.get_next()

# with tf.Session() as sess:
#     # feed the placeholder with data
#     sess.run(iterator.initializer)
#     nxt = sess.run(next)
#     print(type(nxt))
#     # print(sess.run(next))

# Create the BERT model


input_ids = tf.convert_to_tensor(next["input_ids"], name='input_ids')
input_mask = tf.convert_to_tensor(next["input_mask"], name='input_mask')
segment_ids = tf.convert_to_tensor(next["segment_ids"], name='segment_ids')
label_ids = tf.convert_to_tensor(next["label_ids"], name='label_ids')

# input_ids_pl = tf.placeholder(tf.int32, name='input_ids')
# input_mask_pl = tf.placeholder(tf.int32, name='input_mask')
# segment_ids_pl = tf.placeholder(tf.int32, name='segment_ids')
# label_ids_pl = tf.placeholder(tf.int32, name='label_ids')

# total_loss, per_example_loss, logits, probabilities = utils.create_model(
#     bert_config, True, input_ids, input_mask, segment_ids, label_ids,
#     len(label_list), False)

model = bert.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)
output_layer = model.get_pooled_output()

hidden_size = output_layer.shape[-1].value

output_weights = tf.get_variable(
    "output_weights", [len(label_list), hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

output_bias = tf.get_variable(
    "output_bias", [len(label_list)], initializer=tf.zeros_initializer())

with tf.variable_scope("loss"):
    # if is_training:
    if model.is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(label_ids, depth=len(label_list), dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

tvars = tf.trainable_variables()
initialized_variable_names = {}
scaffold_fn = None
if init_checkpoint:
    assignment_map, initialized_variable_names = bert.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

optimizer = optimization.create_optimizer(loss, learning_rate, num_training_steps, 0, False)
init_op = tf.global_variables_initializer()
# Run Tensorflow session
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(1, num_train_epochs + 1):
        sess.run(iterator.make_initializer(dataset))

        while True:
            try:
                _, loss_out, logits_out, probabilities_out = sess.run([optimizer, loss, logits, probabilities])
            except tf.errors.OutOfRangeError:
                break
        print("epoch " + str(epoch))
        print("loss_out" + str(loss_out))
