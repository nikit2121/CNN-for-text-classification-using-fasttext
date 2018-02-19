import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import logging
import seaborn as sns
from matplotlib import pyplot as plt
import re
from tensorflow.contrib import learn
import json
from text_cnn_toxic import TextCNN
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from tensorflow.python import debug as tf_debug


stemr = PorterStemmer()
lmtzr = WordNetLemmatizer()
logging.getLogger().setLevel(logging.INFO)

train_data = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/train/train.csv')
test_data = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/test/test.csv')

"""Check for any null values in test data---- train data does not have any null values"""
print test_data[test_data.comment_text.isnull()]
test_data.comment_text.fillna('empty', inplace=True)

distribution_of_groundTruth = train_data.iloc[:,2:].sum(axis=0)
sns.barplot(distribution_of_groundTruth.index,distribution_of_groundTruth.values)
#plt.show()

"""Function to clean unwanted info from comments"""
STOPWORDS = set(stopwords.words('english'))

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
    s = "".join([i for i in s if i in string.printable])#another try to remove non-ascii characters since the one below didnt work
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


x_raw = train_data['comment_text'].apply(lambda x: clean_str(x)).tolist()

""" Maximum length of a comment then we can pad the shortest one to the longest """
maximum_length = max([len(x.split(' ')) for x in x_raw])
maximum_length_200 =200
logging.info('Maximum length of a comment is :{}'.format(maximum_length))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=maximum_length_200)
x = np.array(list(vocab_processor.fit_transform(x_raw)))
y = np.array(train_data['identity_hate'].tolist())
y = y.transpose()
y = np.resize(y,(len(y),1))

"""Split Train data into 2 parts: train, dev, test"""
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

x_train,x_dev,y_train,y_dev = train_test_split(x_shuffled,y_shuffled,test_size=0.01)

logging.info('x_train: {}, y_train: {}, x_dev: {}, y_dev: {}'.format(len(x_train), len(y_train), len(x_dev), len(y_dev)))

"""Load parameters json file and initialize CNN's hyperparameter"""
parameter_file = '/home/nikit/Desktop/Kaggle/toxic_comments/parameters_toxic.json'
parameter = json.loads(open(parameter_file).read())

"""build the graph and cnn object"""
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=1,
                      vocab_size=len(vocab_processor.vocabulary_),
                      embedding_size=parameter['embedding_dim'],
                      filter_sizes=list(map(int, parameter['filter_sizes'])),
                      num_filters=parameter['num_filters'],
                      l2_reg_lambda=parameter['l2_reg_lambda'])

        global_step = tf.Variable(0,name="global_step",trainable=False)
        learning_rate = tf.train.exponential_decay(0.0001, decay_steps=20,decay_rate= 0.96, staircase=True,global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
        timestamp = str(int(time.time()))

        """set path to save trained model and it's checkpoints"""
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        vocab_processor.save(os.path.join(out_dir,"vocab.pickle"))
        sess.run(tf.global_variables_initializer())
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        #store batches as generator- train_batches
        train_batches = batch_iter(list(zip(x_train,y_train)),parameter['batch_size'], parameter['num_epochs'])
        best_accuracy, best_at_step = 0,0

        """Load pre-trained Fasttext """
        with open('/home/nikit/Desktop/Fasttext/wiki.en.vec') as glove_twitter:
            embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), parameter['embedding_dim']))
            not_missing = []
            for line in glove_twitter:
                value = line.split()
                word = value[0]
                vector = np.asarray(value[1:],dtype="float32")
                idx = vocab_processor.vocabulary_.get(word)
                if idx!=0:
                    embedding_vectors[idx,:] = vector
                    not_missing.append(word)
        glove_twitter.close()
        print("number of missing words in Glove %s"%(len(not_missing)))
        sess.run(cnn.W.assign(embedding_vectors))


        """Take those batches one by one and input to the CNN graph"""
        def train_step(x_batch,y_batch):
            feed_dict = {
                    cnn.input_x:x_batch,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:parameter['dropout_keep_prob']}
            _,step,loss,acc,score = sess.run([train_op,global_step,cnn.loss,cnn.accuracy,cnn.sigmoid_result],feed_dict=feed_dict)
            return acc,loss,score,step


        def dev_step(x_batch,y_batch):
            feed_dict = {
                    cnn.input_x:x_batch,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:1}
            num_correct,acc_,loss_ = sess.run([cnn.num_correct,cnn.accuracy,cnn.loss],feed_dict=feed_dict)
            return num_correct,loss_
        """Train the cnn model using x_train and y_train (batch by batch)"""
        start_time = time.time()
        for train_batch in train_batches:
            x_train_batch, y_train_batch = zip(*train_batch)
            accuracy_,loss_,score_,step_ = train_step(x_train_batch,y_train_batch)
            current_step = tf.train.global_step(sess,global_step)

            dev_batches = batch_iter(list(zip(x_dev,y_dev)),parameter['batch_size'],1)
            total_dev_correct = 0
            for dev_batch in dev_batches:
                x_dev_batch, y_dev_batch = zip(*dev_batch)
                num_correct,loss_ = dev_step(x_dev_batch,y_dev_batch)
                total_dev_correct += num_correct
            dev_accuracy = float(total_dev_correct) / len(y_dev)
            logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))
            logging.critical('Loss on dev set: {}'.format(loss_))

            if dev_accuracy >= best_accuracy:
                best_accuracy, best_at_step = dev_accuracy, current_step
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))
    sess.close()
uni, counts = np.unique(y_dev,return_counts=True)
print counts
print "Hello"
