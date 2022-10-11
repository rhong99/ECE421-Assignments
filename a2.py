%tensorflow_version 1.x
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------ Data processing ------
def loadData():
  with np.load("/content/notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
  return trainData, validData, testData, trainTarget, validTarget, testTarget


def convertOneHot(trainTarget, validTarget, testTarget):
  newtrain = np.zeros((trainTarget.shape[0], 10))
  newvalid = np.zeros((validTarget.shape[0], 10))
  newtest = np.zeros((testTarget.shape[0], 10))

  for item in range(0, trainTarget.shape[0]):
    newtrain[item][trainTarget[item]] = 1
  for item in range(0, validTarget.shape[0]):
    newvalid[item][validTarget[item]] = 1
  for item in range(0, testTarget.shape[0]):
    newtest[item][testTarget[item]] = 1
  return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
  np.random.seed(421)
  randIndx = np.arange(len(trainData))
  target = trainTarget
  np.random.shuffle(randIndx)
  data, target = trainData[randIndx], target[randIndx]
  return data, target


# ------ Part 1: Help Functions ------
def relu(x):
  return np.maximum(0, x)


def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def computeLayer(X, W, b):
  return np.add(np.matmul(X, W), b)


def CE(target, prediction):
  return -np.mean(target * np.log(prediction))


def gradCE(target, prediction):
  return prediction - target


# ------ Part 1: Training NN ------
def train_NN(train_x, valid_x, test_x, train_y, valid_y, test_y,
             epochs, learning_rate, gamma, plot,
             W_hidden, W_v_hidden, b_hidden, b_v_hidden,
             W_outer, W_v_outer, b_outer, b_v_outer):
  
  print("------------ Part 1 ------------")

  # Accuracies and losses
  train_acc = []
  train_loss = []
  valid_acc = []
  valid_loss = []
  test_acc = []
  test_loss = []
  epochnum = []

  # Set time
  orig_time = time.time()

  # Training loop
  for epoch in range(0, epochs):

    epochnum.append(epoch)

    # Training
    hidden_in = computeLayer(train_x, W_hidden, b_hidden)
    hidden = relu(hidden_in)
    prediction = softmax(computeLayer(hidden, W_outer, b_outer))
    train_loss.append(CE(train_y, prediction))
    compare = np.equal(np.argmax(prediction, axis=1),
                       np.argmax(train_y, axis=1))
    accuracy = np.sum((compare==True)) / (len(train_y))
    train_acc.append(accuracy)

    # Save for validation and testing
    W_hidden_old = W_hidden
    b_hidden_old = b_hidden
    W_outer_old = W_outer
    b_outer_old = b_outer

    # Back propogation and update
    # Output
    W_outer_grad = np.matmul(np.transpose(hidden),
                              gradCE(train_y, prediction))
    W_v_outer = (gamma * W_v_outer) + (learning_rate * W_outer_grad)
    W_outer = W_outer - W_v_outer
    b_outer_grad = np.matmul(np.ones((1, train_y.shape[0])),
                              gradCE(train_y, prediction))
    b_v_outer = (gamma * b_v_outer) + (learning_rate * b_outer_grad)
    b_outer = b_outer - b_v_outer

    # Hidden
    hidden_in[hidden_in > 0] = 1
    hidden_in[hidden_in < 0] = 0
    W_hidden_grad = np.matmul(np.transpose(train_x),
                              (hidden_in *
                               np.matmul(gradCE(train_y, prediction),
                                         np.transpose(W_outer))))
    W_v_hidden = (gamma * W_v_hidden) + (learning_rate * W_hidden_grad)
    W_hidden = W_hidden - W_v_hidden
    b_hidden_grad = np.matmul(np.ones((1, hidden_in.shape[0])),
                              (hidden_in *
                               np.matmul(gradCE(train_y, prediction),
                                         np.transpose(W_outer))))
    b_v_hidden = (gamma * b_v_hidden) + (learning_rate * b_hidden_grad)
    b_hidden = b_hidden - b_v_hidden

    # Validation
    hidden_in = computeLayer(valid_x, W_hidden_old, b_hidden_old)
    hidden = relu(hidden_in)
    prediction = softmax(computeLayer(hidden, W_outer_old, b_outer_old))
    valid_loss.append(CE(valid_y, prediction))
    compare = np.equal(np.argmax(prediction, axis=1),
                       np.argmax(valid_y, axis=1))
    accuracy = np.sum((compare==True)) / (len(valid_y))
    valid_acc.append(accuracy)

    # Testing
    hidden_in = computeLayer(test_x, W_hidden_old, b_hidden_old)
    hidden = relu(hidden_in)
    prediction = softmax(computeLayer(hidden, W_outer_old, b_outer_old))
    test_loss.append(CE(test_y, prediction))
    compare = np.equal(np.argmax(prediction, axis=1),
                       np.argmax(test_y, axis=1))
    accuracy = np.sum((compare==True)) / (len(test_y))
    test_acc.append(accuracy)

    # Print
    if epoch % 10 == 0:
      runtime = time.time() - orig_time
      print("Epoch: {}  Time Elapsed: {b:.3f}s".format(epoch, b=runtime))
      print("Training")
      print("Accuracy: {a:.5f}  Loss: {b:.5f}".format(a=train_acc[-1], b=train_loss[-1]))
      print("Validation")
      print("Accuracy: {a:.5f}  Loss: {b:.5f}".format(a=valid_acc[-1], b=valid_loss[-1]))
      print(" ")

  runtime = time.time() - orig_time

  print("------------ Finished Training ------------")
  print("Epoch: {}  Time Elapsed: {b:.3f}s".format(epochs, b=runtime))
  print("Training")
  print("Accuracy: {a:.5f}  Loss: {b:.5f}".format(a=train_acc[-1], b=train_loss[-1]))
  print("Validation")
  print("Accuracy: {a:.5f}  Loss: {b:.5f}".format(a=valid_acc[-1], b=valid_loss[-1]))
  print("Testing")
  print("Accuracy: {a:.5f}  Loss: {b:.5f}".format(a=test_acc[-1], b=test_loss[-1]))
  print(" ")
  print(" ")

  # Plotting
  if plot == True:
    plot_graphs(train_acc, valid_acc, test_acc, epochnum, 1)        #accuracy plot
    plot_graphs(train_loss, valid_loss, test_loss, epochnum, 0)     #loss plot

  return W_hidden, b_hidden, W_outer, b_outer

# ------ Part 2: Model ------

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], \
        padding='SAME', use_cudnn_on_gpu=True)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], \
        strides=[1, k, k, 1],padding='SAME')

def conv_model(x, weights, biases, keep_prob):

  conv1 = conv2d(x, weights['w1'], biases['b1'])
    
  #batch normalization layer
  mean, variance = tf.nn.moments(conv1, [0,1,2])
  scale = tf.Variable(tf.ones([32]))
  beta = tf.Variable(tf.ones([32]))
  batch_norm = tf.nn.batch_normalization(conv1, mean, variance, beta, scale, 1e-3)

  #max pooling
  pooling = maxpool2d(batch_norm, 2)

  # Reshape conv2 output
  fc1 = tf.reshape(pooling, [-1, weights['w2'].get_shape().as_list()[0]])
  fc1 = tf.add(tf.matmul(fc1, weights['w2']), biases['b2'])
    
  # ReLU activation
  fc1 = tf.nn.relu(fc1)

  # apply dropout
  fc1 = tf.nn.dropout(fc1, keep_prob) 
    
  # Fully connected layer
  fc2 = tf.reshape(fc1, [-1, weights['w3'].get_shape().as_list()[0]])
  fc2 = tf.add(tf.matmul(fc2, weights['w3']), biases['b3'])
    
  # Output layer
  output = tf.add(tf.matmul(fc2, weights['w_out']), biases['b_out'])

  return output


def train_nn_part2(train_x, valid_x, test_x, train_y, valid_y, 
                   test_y, epochs, learning_rate, batch_size, lamda, plot):

  tf.reset_default_graph()

  x = tf.placeholder("float", [None, 28,28,1])
  y = tf.placeholder("float", [None, 10])
  keep_prob = tf.placeholder(tf.float32)
  
  weights = {
    'w1': tf.get_variable('W00', shape=(4,4,1,32), \
        initializer=tf.contrib.layers.xavier_initializer()), 
    'w2': tf.get_variable('W11', shape=(14*14*32,64), \
        initializer=tf.contrib.layers.xavier_initializer()),  
    'w3': tf.get_variable('W33', shape=(64,128), \
        initializer=tf.contrib.layers.xavier_initializer()), 
    'w_out': tf.get_variable('W66', shape=(128,10), \
        initializer=tf.contrib.layers.xavier_initializer()), 
  }
  biases = {
    'b1': tf.get_variable('B00', shape=(32), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'b2': tf.get_variable('B11', shape=(64), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'b3': tf.get_variable('B22', shape=(128), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'b_out': tf.get_variable('B44', shape=(10), \
        initializer=tf.contrib.layers.xavier_initializer()),
  } 

  pred = conv_model(x, weights, biases, keep_prob)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
  optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost) 

# THIS IS FOR 2.3.1 HYPERPARAMETER INVESTIGATION 
  regularizers = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + \
  tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w_out'])
  cost = tf.reduce_mean(cost + lamda * regularizers)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

  correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    batch_num = int(len(train_x)//batch_size)
    
    # Accuracies and losses
    trainacc = []
    trainloss = []
    validacc = []
    validloss = []
    testacc = []
    testloss = []
    epochnum = []

    train_x = train_x.reshape(-1, 28, 28, 1)
    valid_x = valid_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)

    orig_time = time.time()

    for i in range(epochs):

      epochnum.append(i)
      train_x, train_y = shuffle(train_x, train_y)
      onehot_train, onehot_valid, onehot_test = convertOneHot(train_y, valid_y, test_y)

      for j in range(0, batch_num):
        batch_x = train_x[j * batch_size : min((j + 1) * batch_size, len(train_x))]
        batch_y = onehot_train[j * batch_size : min((j + 1) * batch_size, len(onehot_train))]

        opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5 })
      
      tr_loss, tr_acc = sess.run([cost, accuracy], feed_dict={x: train_x, y: onehot_train, keep_prob: 0.5})
      te_acc,te_loss = sess.run([accuracy,cost], feed_dict={x: test_x, y : onehot_test, keep_prob: 0.5 })
      v_acc,v_loss = sess.run([accuracy,cost], feed_dict={x: valid_x,y : onehot_valid, keep_prob: 0.5 })

      trainloss.append(tr_loss)
      validloss.append(v_loss)
      testloss.append(te_loss)
        
      trainacc.append(tr_acc)
      validacc.append(v_acc)
      testacc.append(te_acc)

      if i % 10 == 0:
        runtime = time.time() - orig_time
        print("Epoch: {}  Time Elapsed: {b:.3f}s".format(i, b=runtime))
        print("Training Accuracy: {a:.5f}  Validation Accuracy: {b:.5f}".format(a=trainacc[-1], b=validacc[-1]))
        print("Training Loss: {a:.5f}  Validation Loss: {b:.5f}".format(a=trainloss[-1], b=validloss[-1]))
        print(" ")

    runtime = time.time() - orig_time

    if plot == True:
      plot_graphs(trainacc, validacc, testacc, epochnum, 1)
      plot_graphs(trainloss, validloss, testloss, epochnum, 0)

  return trainloss, validloss

def plot_graphs(training, validation, test, epochnum, is_accuracy):
    plt.plot(epochnum, training, label = 'training data')
    plt.plot(epochnum, validation, label = 'validation data')
    plt.plot(epochnum, test, label = 'test data')
    if (is_accuracy == 1):
        plt.title('Accuracy vs. Number of Epoch')
        plt.xlabel("Number of Epoch")
        plt.ylabel("Accuracy")
        plt.legend(('Training', 'Validation','test'))
    if (is_accuracy == 0):
        plt.title('Loss vs. Number of Epoch')
        plt.xlabel("Number of Epoch")
        plt.ylabel("Loss")
        plt.legend(('Training', 'Validation','test'))
    plt.show()

# ------ Main ------
def main():
  print("Tensorflow version: ", tf.__version__)
  print(" ")

  # Settings
  part_1 = False
  part_2 = True
  plot = False

  # Hyperparameters
  epochs = 50
  learning_rate = 1e-4
  hidden_units = 1000
  gamma = 0.99
  batch_size = 32
  lamda = 0.01    # FOR 2.3.1 HYPERPARAMETER INVESTIGATION, TEST WITH 0.01, 0.1, 0.5

  # Split data
  train_x, valid_x, test_x, train_y, valid_y, test_y = loadData()

  train_x = train_x.reshape((len(train_y), -1))
  valid_x = valid_x.reshape((len(valid_y), -1))
  test_x = test_x.reshape((len(test_y), -1))

  train_y_onehot, valid_y_onehot, test_y_onehot = convertOneHot(train_y, valid_y, test_y)


  # Initialize parameters
  # Hidden
  mean = 0
  std_dev = 1.0 / (len(train_y) + hidden_units)
  W_hidden = np.random.normal(mean, np.sqrt(std_dev), (train_x.shape[1], hidden_units))
  W_v_hidden = np.full((train_x.shape[1], hidden_units), 1e-5)
  b_hidden = np.zeros((1, hidden_units))
  b_v_hidden = np.zeros((1, hidden_units))

  # Outer
  mean = 0
  std_dev = 1.0 / (hidden_units + 10)
  W_outer = np.random.normal(mean, np.sqrt(std_dev), (hidden_units, 10))
  W_v_outer = np.full((hidden_units, 10), 1e-5)
  b_outer = np.zeros((1, 10))
  b_v_outer = np.zeros((1, 10))

  # Training
  if part_1 is True:
    W_hidden_new, b_hidden_new, W_outer_new, b_outer_new = train_NN(
        train_x, valid_x, test_x, train_y_onehot, valid_y_onehot, test_y_onehot,
        epochs, learning_rate, gamma, plot,
        W_hidden, W_v_hidden, b_hidden, b_v_hidden,
        W_outer, W_v_outer, b_outer, b_v_outer)
    
  if part_2 is True:
    trainloss, validloss = train_nn_part2(
        train_x, valid_x, test_x, train_y, valid_y, test_y,
        epochs, learning_rate, batch_size, lamda, plot)


  
