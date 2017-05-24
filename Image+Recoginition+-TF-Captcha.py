
# coding: utf-8

# In[10]:

import tensorflow as tf
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import PIL.Image
from os.path import isfile, join, exists
from os import listdir
from sklearn import preprocessing
from random import randint


# In[2]:

image_base_path = 'captcha/images'
test_image_path = 'test'
train_image_path = 'train'
validate_image_path = 'validate'
image_extension = 'png'


# In[4]:

def one_hot_encoding(data):
    lb = preprocessing.LabelBinarizer()
    lb.fit([0,1,2,3,4,5,6,7,8,9])
    return lb.transform(data)


# In[5]:

def image_to_array(complete_path):
    img = PIL.Image.open(open(complete_path,"rb"))
    mat = np.array(img)
    flat_arr = mat.ravel()
    vector = np.array(flat_arr)
    return vector
    


# In[7]:

def create_image_path(base,specific,file_name,extension):
    return base + '/'+specific+'/'+file_name


# In[8]:

def create_directory_path(base,specific):
    return base + '/'+specific


# In[10]:

def extract_label_from_filename(file_name):
    return int(file_name.split('_')[1].split('.')[0])
    
    


# In[12]:

def read_all_files_from_directory(specific):
    
    directory = create_directory_path(base=image_base_path,specific=specific)
    if exists(directory):
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
        return onlyfiles
    return None


# In[13]:

def load_data(specific):

    input_arr = np.empty((0,10800),int)
    label_arr = np.empty((0), dtype=int)
    onlyfiles = read_all_files_from_directory(specific=specific)
    if onlyfiles is not None:
        for i in range(len(onlyfiles)):
            if 'png' in onlyfiles[i]:
                
                temp_input = image_to_array(create_image_path(image_base_path,specific,onlyfiles[i],image_extension))
                temp_label = extract_label_from_filename(onlyfiles[i])
                
                input_arr = np.append(input_arr,np.array([temp_input]),axis=0) 
                label_arr = np.append(label_arr,temp_label)
    return input_arr,label_arr            
                


# Below step is quite slow for not , it will take atleas 5 mins

# In[14]:

train_input , train_label =  load_data(train_image_path)
print('training data loaded')
validate_input,validate_label = load_data(validate_image_path)
print('validating data loaded')
test_input , test_label = load_data(test_image_path)
print('test data loaded')


# In[15]:

train_label_en = one_hot_encoding(train_label)
validate_label_en = one_hot_encoding(validate_label)
test_label_en = one_hot_encoding(test_label)


# In[16]:

print('input data shape',train_input.shape,train_input[0].shape)
print('label data shape',train_label.shape,train_label[0].shape)


# In[14]:

filter_size_1 = 15
filter_num_1 = 16

filter_size_2 = 12
filter_num_2 = 32

fc_size_0 = 4000
fc_size_1 = 2000
fc_size_2 = 1000
fc_size_3 = 200


# In[2]:

img_size = 60
num_channels = 3
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size,num_channels)
num_classes = 10


# In[15]:

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[16]:

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# In[17]:

def new_biases(length):
    #equivalent to y intercept
    #constant value carried over across matrix math
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[18]:

def new_con_layer(input,number_input_channel,filter_size,number_of_filters,use_pooling= True):
    
    shape = [filter_size, filter_size, number_input_channel, number_of_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=number_of_filters)
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
    
    if use_pooling:
        layer = tf.nn.max_pool(layer,strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME')
        
    layer = tf.nn.relu(layer)
    return layer,weights


# In[19]:

def flatten_layer(input):
    input_shape = input.shape
    num_features = input_shape[1:4].num_elements()
    layer_flat = tf.reshape(input, [-1, num_features])
    return layer_flat,num_features


# In[20]:

def fully_connected_layer(input,num_of_input,num_of_output,use_relu =True):
    
    weights = new_weights(shape=[num_of_input, num_of_output])
    biases = new_biases(length=num_of_output)
    layer = tf.matmul(input,weights)+biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# In[21]:

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# In[22]:

layer_conv1, weights_conv1 =     new_con_layer(input=x_image,
                   number_input_channel=num_channels,
                   filter_size=filter_size_1,
                   number_of_filters=filter_num_1,
                   use_pooling=True)
print(layer_conv1)    


# In[23]:

layer_conv2, weights_conv2 =     new_con_layer(input=layer_conv1,
                   number_input_channel=filter_num_1,
                   filter_size=filter_size_2,
                   number_of_filters=filter_num_2,
                   use_pooling=True)
print(layer_conv2)    


# In[24]:

layer_flatten,number_of_features = flatten_layer(layer_conv2)
print(layer_flatten)
print(number_of_features)


# In[27]:

layer_fc1 = fully_connected_layer(layer_flatten,number_of_features,fc_size_0)
print(layer_fc1)

layer_fc1 = fully_connected_layer(layer_fc1,fc_size_0,fc_size_1)
print(layer_fc1)
layer_fc1 = fully_connected_layer(layer_fc1,fc_size_1,fc_size_2)
print(layer_fc1)
layer_fc1 = fully_connected_layer(layer_fc1,fc_size_2,fc_size_3)
print(layer_fc1)


# In[28]:

layer_fc2 = fully_connected_layer(layer_fc1,fc_size_3,num_classes,use_relu=False)
print(layer_fc2)


# In[29]:

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[66]:


session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 64
total_iterations = 0
num_iterations = 10000


# In[67]:

def getBatch(data,label,batch_size):
    data_size = len(data)
    lower_limit = randint(0,data_size-batch_size)
    return data[lower_limit:lower_limit+batch_size],label[lower_limit:lower_limit+batch_size]
    


# In[68]:


def optimize():
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = getBatch(train_input,train_label_en,train_batch_size)
        #data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# In[69]:

def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = train_label[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# In[70]:

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[71]:

test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(test_input)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = test_input[i:j, :]

        # Get the associated labels.
        labels = test_label_en[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = train_label
    print('cls_true',cls_true)
    print('cls shape',cls_true.shape)
    print('cls perd',cls_pred)
    print('cls perd shape',cls_pred.shape)    
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = np.count_nonzero(correct)#correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# In[72]:

print_test_accuracy()


# In[73]:

optimize()

