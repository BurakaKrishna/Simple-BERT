# Simplified version of Bidirectional Encoder Representations from Transformers (BERT)

## Overview
This is a modified version of the 
[Bidirectional Encoder Representations from Transformers (BERT)](https://github.com/google-research/bert)
from google research.

The main purpose is to allow for periodic evaluation of the model during training while using a custom estimator.
Something that was surprisingly lacking in the original code.
This code will evaluate the model accuracy and loss, on both training and evaluation sets,
after a specified number of seconds (evaluate_secs) and record the results to tensorboard.
In the process several other evaluation metrics have been added for the test dataset,
including a confusion matrix, F1, Precision and Recall.

This code is also intended to make it more clear how the BERT model is created and trained for those who are not
familiar with Tensorflow Estimators.

## Datasets
Currently there is only two datsets included.
The [Switchboard Dialogue Act Corpus (SWDA)](https://github.com/NathanDuran/Switchboard-Corpus)
and the [Meeting Recorder Dialogue Act Corpus (MRDA)](https://github.com/NathanDuran/MRDA-Corpus).
**Note:** If you want to use the MRDA corpus you need to specify which label types you want to use within the 
MrdaProcessor() function. For more information on the label types see the above MRDA link.

However, to run this code on any of the other datasets included with the original BERT model you can
simply process and run them in the same way described in the original documentation.

## Usage
You must include one of the original pre-trained BERT models in the root directory (BERT_Base or BERT_Large).
Due to the model checkpoint size they have not been included in this repository.

Currently the model will save the single best checkpoint (lowest loss) to the best_checkpoint directory.

Within the run_bert.py script you must specify the following parameters.
- task_name = Which task (DataProcessor) to use
- experiment_name = An (optional) parameter for running different experiments on the same dataset
- max_seq_length = Maximum sentence/sequence length (dependent on pre-trained model)
- batch_size = How large each training batch should be (default 32)
- learning_rate = Model learning rate (default 2e-5)
- num_epochs = Number of times to iterate over training data (default 3)
- save_checkpoint_steps = Number of global steps to run before saving a checkpoint
- evaluate_secs = After how many seconds to evaluate the model
- checkpoints_to_keep = Number of 'best' checkpoints to keep
- training = Boolean flag whether to train the model
- testing = Boolean flag whether to test the model after training
- bert_model_type = Directory of the pre-trained BERT model (i.e Base or Large)
- do_lower_case = Depending on the pre-trained model you are using

The run_bert_no_estimator.py script runs the BERT model without estimators entirely,
however this is inconsistent and untested.
This is because the original BERT code does not allow easy switching between using dropout for training
and not using it for evaluation. **Use at your own risk!**

## TODO
- Add command line support for scripts.
- Add option to extract features rather than make predictions (like BERT's extract_features.py).

## Run Tensorboard
tensorboard --logdir=./<output_dir>

## Acknowledgments
Thanks to bluecamel for the [Best Checkpoint Copier](https://github.com/bluecamel/best_checkpoint_copier) code.