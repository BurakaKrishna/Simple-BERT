import os
import collections
import csv
import pickle
import tokenization
import tensorflow as tf
# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_eval_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the eval set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SwdaProcessor(DataProcessor):
    """Processor for the Switchboard data set."""

    def get_train_examples(self, data_dir):
        """Training Set."""
        with open(data_dir + 'train_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "train")

    def get_eval_examples(self, data_dir):
        """Evaluation Set.
        Set here WILL have labels and be used to evaluate training"""
        with open(data_dir + 'eval_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "eval")

    def get_test_examples(self, data_dir):
        """Test Set.
        Set here will NOT have labels and be used to make predictions"""
        with open(data_dir + 'test_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "test")

    def get_labels(self, data_dir):
        """Load labels from pickled metadata."""
        with open(data_dir + '/metadata/metadata.pkl', 'rb') as handle:
            metadata = pickle.load(handle)
        return list(metadata['label_freq'].keys())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # Need to split lines on '|' character
        sentences = []
        labels = []
        for line in lines:
            sentences.append(line.split("|")[1])
            labels.append(line.split("|")[2])
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line.split("|")[1])
            label = tokenization.convert_to_unicode(line.split("|")[2])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MrdaProcessor(DataProcessor):
    """Processor for the Switchboard data set."""

    def get_train_examples(self, data_dir):
        """Training Set."""
        with open(data_dir + 'train_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "train")

    def get_eval_examples(self, data_dir):
        """Evaluation Set.
        Set here WILL have labels and be used to evaluate training"""
        with open(data_dir + 'eval_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "eval")

    def get_test_examples(self, data_dir):
        """Test Set.
        Set here will NOT have labels and be used to make predictions"""
        with open(data_dir + 'test_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "test")

    def get_labels(self, data_dir):
        """Load labels from pickled metadata."""
        with open(data_dir + '/metadata/metadata.pkl', 'rb') as handle:
            metadata = pickle.load(handle)
        # basic_label_freq, general_label_freq or full_label_freq
        return list(metadata['full_label_freq'].keys())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # Need to split lines on '|' character
        sentences = []
        labels = []
        for line in lines:
            sentences.append(line.split("|")[1])
            labels.append(line.split("|")[4])  # Index 2 = basic, 3 = general and 4 = full
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line.split("|")[1])
            label = tokenization.convert_to_unicode(line.split("|")[4])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    # if ex_index < 5:
    #     tf.logging.info("*** Example ***")
    #     tf.logging.info("guid: %s" % example.guid)
    #     tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
    #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            int_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return int_feature

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def build_dataset(input_file, seq_length, batch_size, is_training=True, drop_remainder=False):
    """Creates an iterable dataset for BERT from the specified TF Record File"""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    dataset = tf.data.TFRecordDataset(input_file)
    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=100)

    dataset = dataset.map(lambda record: tf.parse_single_example(record, name_to_features))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
