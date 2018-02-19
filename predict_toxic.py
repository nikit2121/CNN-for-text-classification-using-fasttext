import os
import csv
import json
import logging
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

stemr = PorterStemmer()
lmtzr = WordNetLemmatizer()

logging.getLogger().setLevel(logging.INFO)
STOPWORDS = set(stopwords.words('english'))

"""Function to clean unwanted info from comments"""
def clean_str(s):
    """Clean sentence"""

    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    #s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " had / would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"'m"," am",s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(" \d+", "", s)
    s = [word for word in word_tokenize(s.lower()) if word not in STOPWORDS]
    s = [lmtzr.lemmatize(i) for i in s]
    s = " ".join(s)
    s = s.strip()
    return s
"""Create batches using below function batch_iter"""
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size)+1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def predict_unseen_data():
    """Load parameters json file and initialize CNN's hyperparameter"""
    parameter_file = '/home/nikit/Desktop/Kaggle/toxic_comments/parameters_toxic.json'
    parameter = json.loads(open(parameter_file).read())
    checkpoint_dir = '/home/nikit/Desktop/Kaggle/toxic_comments/trained_model_1517892368/'
    if not checkpoint_dir.endswith('/'):
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

    """Step 1: load data for prediction"""
    test_data = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/test/test.csv')
    test_data.comment_text.fillna('empty', inplace=True)
    x_raw = test_data['comment_text'].apply(lambda x: clean_str(x)).tolist()
    # labels.json was saved during training, and it has to be loaded during prediction

    logging.info('The number of x_test: {}'.format(len(x_raw)))
    vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    """Step 2: compute the predictions"""
    value = True
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            sigmoid = graph.get_operation_by_name("output/sigmod").outputs[0]
            h_pool = graph.get_operation_by_name("h_pool/h_pool_flat").outputs[0]
            #predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            #h_pool_flat = graph.get_operation_by_name("h_pool/h_pool_flat").outputs[0]

            batches = batch_iter(list(x_test), parameter['batch_size'], 1, shuffle=False)
            all_scores = []
            for x_test_batch in batches:
                b_s,flat = sess.run([sigmoid,h_pool], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                for index,i in enumerate(b_s):
                    all_scores.append(i)


        sess.close()
    np.savetxt('/home/nikit/Desktop/Kaggle/toxic_comments/results/eight/predictions6.csv',all_scores)
if __name__ == '__main__':
    predict_unseen_data()
