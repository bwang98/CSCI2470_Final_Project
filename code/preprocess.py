import pandas as pd
import numpy as np
import tensorflow as tf

# replace the file directory with yours.
df = pd.read_csv("data/fer2013.csv")
# Combining PublicTest and PrivateTest to the whole testing set
df_training = df[df["Usage"] == "Training"]
df_testing = df[df["Usage"] != "Testing"]
# Save training set and testing set as csv file for the purpose of futher use
# Uncomment commands below if you need to save these files
#df_training.to_csv("../data/train.csv", index = False)
#df_testing.to_csv("../data/test.csv", index = False)



# create inputs and labels for the training set
def get_inputs_labels(file_path):
    """
    This function is used to get inputs and labels for the csv file described
    by the file path.

    :param file_path: the file directory
    :return: numpy array of inputs and numpy array of labels
    """
    df = pd.read_csv(file_path)
    inputs = [np.array(df["pixels"][i].split(" ")) for i in range(df.shape[0])]
    labels = [df["emotion"][i] for i in range(df.shape[0])]
    assert len(inputs) == len(labels), "inputs and labels should have the same length."
    return np.array(inputs), np.array(labels)


def training_testing_prep(training_file_path, testing_file_path):
    """
    Given the training_file_path and testing_file_path, this function will return reshaped 
    training pixels, the one-hot encoder of training labels, reshaped testing pixels, and
    the one-hot encoder of testing labels 
    
    :param training_file_path: the training set path
    :param testing_file_path: the tsting set path
    :return: training_inputs in the shape of [training_set size, image_size, image_size, channel_size]
    :return: training_labels after one-hot encoding
    :return: testing_labels after one-hot encoding
    """
    image_size = 48
    training_inputs, training_labels = get_inputs_labels(training_file_path)
    testing_inputs, testing_labels = get_inputs_labels(testing_file_path)
    training_inputs = training_inputs.reshape(training_inputs.shape[0], image_size, image_size, 1)
    testing_inputs = testing_inputs.reshape(testing_inputs.shape[0], image_size, image_size, 1)
    training_labels = tf.one_hot(training_labels, 7, dtype=tf.float32)
    testing_labels = tf.one_hot(testing_labels, 7, dtype=tf.float32)
    
    return training_inputs, training_labels, testing_inputs, testing_labels




