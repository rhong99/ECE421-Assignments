# ECE421 - Assignment 1
# Ryan Hong (1003960822)
# Jueun Lee (1004018939)

%tensorflow_version 1.x
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

print("Tensorflow version: ", tf.__version__)
print(" ")


# Loading data into training, validation, and testing sets


def loadData():
    with np.load('/content/notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Part 1 - Linear Regression (MSE)


def MSE(W, b, x, y, reg):
    y_hat = np.matmul(x, W) + b
    e = y_hat - y
    # loss = (np.sum(e * e) / (2 * len(y))) 
    loss = (np.sum(e * e) / (2 * len(y))) + (reg * np.sum(W * W) / 2)
    return loss


def gradMSE(W, b, x, y, reg):
    y_hat = np.matmul(x, W) + b
    e = y_hat - y
    # dloss_dw = np.matmul(np.transpose(x),e)/(len(y)) 
    dloss_dw = np.matmul(np.transpose(x),e)/(len(y)) + (2 * reg * W)
    dloss_db = (np.sum(e)) / (len(y))
    return dloss_dw, dloss_db


def grad_descent(W, b, train_x, train_y, valid_x, valid_y, test_x, test_y, alpha, epochs, reg, error_tol):
    print("---------MSE Gradient Descent---------")

    train_x = train_x.reshape(len(train_y), 28 * 28)
    valid_x = valid_x.reshape(len(valid_y), 28 * 28)
    test_x = test_x.reshape(len(test_y), 28 * 28)

    trainacc = []
    trainloss = []
    validacc = []
    validloss = []
    testacc = []
    testloss = []
    epochnum = []

    orig_time = time.time()

    for i in range(0, epochs):

        # Calculate the gradient
        dloss_dw, dloss_db = gradMSE(W, b, train_x, train_y, reg)

        W_new = W - (alpha * dloss_dw)
        b_new = b - (alpha * dloss_db)

        # Compute loss
        loss = MSE(W, b, train_x, train_y, reg)
        trainloss.append(loss)
        loss = MSE(W, b, valid_x, valid_y, reg)
        validloss.append(loss)
        loss = MSE(W, b, test_x, test_y, reg)
        testloss.append(loss)

        # Compute accuracy
        acc = accuracy(W, b, train_x, train_y)
        trainacc.append(acc)
        acc = accuracy(W, b, valid_x, valid_y)
        validacc.append(acc)
        acc = accuracy(W, b, test_x, test_y)
        testacc.append(acc)

        epochnum.append(i)

        # Check error_tol
        if np.linalg.norm(W_new - W) < error_tol:
            return W_new, b_new

        W = W_new
        b = b_new

        if i % 100 == 0:
            runtime = time.time() - orig_time
            print("Epoch: {}  Time Elapsed: {b:.3f}s".format(i, b=runtime))
            print("Training Accuracy: {a:.5f}  Validation Accuracy: {b:.5f}".format(a=trainacc[-1], b=validacc[-1]))
            print("Training Loss: {a:.5f}  Validation Loss: {b:.5f}".format(a=trainloss[-1], b=validloss[-1]))
            print(" ")

    runtime = time.time() - orig_time

    print("-----Finished Training-----")
    print(" ")
    print("Epoch: {}  Time Elapsed: {b:.3f}s".format(epochs, b=runtime))
    print("Training Accuracy: {a:.5f}  Validation Accuracy: {b:.5f}".format(a=trainacc[-1], b=validacc[-1]))
    print("Training Loss: {a:.5f}  Validation Loss: {b:.5f}".format(a=trainloss[-1], b=validloss[-1]))
    print(" ")
    print("Testing Data Set")
    print("Testing Accuracy: {a:.5f}  Testing Loss: {b:.5f}".format(a=testacc[-1], b=testloss[-1]))
    print(" ")

    # plot(trainacc, validacc, testacc, epochnum, 1)
    # plot(trainloss, validloss, testloss, epochnum, 0)

    return W, b, trainloss, validloss, testloss, epochnum


# Part 2 - Logistic Regression (CE)


def crossEntropyLoss(W, b, x, y, reg, epsilon):
    y_hat = 1 / (1 + np.exp(-(np.matmul(x, W)) + b))
    # Using epsilon = 1e-5 to solve division by 0 problem
    loss = ((np.sum(-((y * np.log(y_hat + epsilon)) + ((1 - y) * np.log(1 - y_hat + epsilon))))) / (len(y)))
    with_reg = (reg * np.sum(W * W) / 2)
    loss = loss + with_reg
    return loss


def gradCE(W, b, x, y, reg):
    y_hat = 1 / (1 + np.exp(-(np.matmul(x, W)) + b))
    dloss_dw = (np.matmul(np.transpose(x), (y_hat - y)) / (len(y))) + (2 * reg * W)
    dloss_db = np.sum((y_hat - y)) / len(y)
    return dloss_dw, dloss_db


def grad_descent_CE(W, b, train_x, train_y, valid_x, valid_y, test_x, test_y, alpha, epochs, reg, error_tol, epsilon):
    print("---------Cross Entropy Gradient Descent---------")

    train_x = train_x.reshape(len(train_y), 28 * 28)
    valid_x = valid_x.reshape(len(valid_y), 28 * 28)
    test_x = test_x.reshape(len(test_y), 28 * 28)

    trainacc = []
    trainloss = []
    validacc = []
    validloss = []
    testacc = []
    testloss = []
    epochnum = []

    orig_time = time.time()

    for i in range(0, epochs):
        # Calculate the gradient
        dloss_dw, dloss_db = gradCE(W, b, train_x, train_y, reg)

        W_new = W - (alpha * dloss_dw)
        b_new = b - (alpha * dloss_db)

        # Compute loss
        loss = crossEntropyLoss(W, b, train_x, train_y, reg, epsilon)
        trainloss.append(loss)
        loss = crossEntropyLoss(W, b, valid_x, valid_y, reg, epsilon)
        validloss.append(loss)
        loss = crossEntropyLoss(W, b, test_x, test_y, reg, epsilon)
        testloss.append(loss)

        # Compute accuracy
        acc = accuracy(W, b, train_x, train_y)
        trainacc.append(acc)
        acc = accuracy(W, b, valid_x, valid_y)
        validacc.append(acc)
        acc = accuracy(W, b, test_x, test_y)
        testacc.append(acc)

        epochnum.append(i)

        # Check error_tol
        if np.linalg.norm(W_new - W) < error_tol:
            return W_new, b_new

        W = W_new
        b = b_new

        if i % 100 == 0:
            runtime = time.time() - orig_time

            print("Epoch: {}  Time Elapsed: {b:.3f}s".format(i, b=runtime))
            print("Training Accuracy: {a:.5f}  Validation Accuracy: {b:.5f}".format(a=trainacc[-1], b=validacc[-1]))
            print("Training Loss: {a:.5f}  Validation Loss: {b:.5f}".format(a=trainloss[-1], b=validloss[-1]))
            print(" ")

    runtime = time.time() - orig_time

    print("-----Finished Training-----")
    print(" ")
    print("Epoch: {}  Time Elapsed: {b:.3f}s".format(epochs, b=runtime))
    print("Training Accuracy: {a:.5f}  Validation Accuracy: {b:.5f}".format(a=trainacc[-1], b=validacc[-1]))
    print("Training Loss: {a:.5f}  Validation Loss: {b:.5f}".format(a=trainloss[-1], b=validloss[-1]))
    print(" ")
    print("Testing Data Set")
    print("Testing Accuracy: {a:.5f}  Testing Loss: {b:.5f}".format(a=testacc[-1], b=testloss[-1]))
    print(" ")

    # plot(trainacc, validacc, testacc, epochnum, 1)
    # plot(trainloss, validloss, testloss, epochnum, 0)

    return W, b, trainloss, validloss, testloss, epochnum


# Part 3 - Batch GD vs. SGD and Adam


def buildGraph(loss_function, batch_size, lr, ep, beta_mse, beta_ce, epochs):
    #Initialize weight and bias tensors
    tf.set_random_seed(421)
    g = tf.Graph()
    batch_size = 500
    lr = 0.001
    ep = 1e-5
    beta_mse = 0
    beta_ce = 0
    epochs = 2500


    with g.as_default():
        W = tf.Variable(tf.random_normal(shape=(28*28, 1), dtype=tf.float32))
        b = tf.Variable(tf.zeros(1))

        train_x = tf.placeholder(tf.float32, shape=(batch_size, 28*28))
        train_y = tf.placeholder(tf.float32, shape=(batch_size, 1))
        valid_x = tf.placeholder(tf.float32, shape=(100, 28*28))
        valid_y = tf.placeholder(tf.float32, shape=(100, 1))
        test_x = tf.placeholder(tf.float32, shape=(145, 28*28))
        test_y = tf.placeholder(tf.float32, shape=(145, 1))

        if loss_function == "MSE":
            train_y_hat = tf.matmul(train_x, W) + b
            loss = tf.losses.mean_squared_error(train_y, train_y_hat)
            reg = tf.nn.l2_loss(W)
            trainingloss = loss + (beta_mse * reg / 2)

            valid_y_hat = tf.matmul(valid_x, W) + b
            loss = tf.losses.mean_squared_error(valid_y, valid_y_hat)
            reg = tf.nn.l2_loss(W)
            validationloss = loss + (beta_mse * reg / 2)

            test_y_hat = tf.matmul(test_x, W) + b
            loss = tf.losses.mean_squared_error(test_y, test_y_hat)
            reg = tf.nn.l2_loss(W)
            testingloss = loss + (beta_mse * reg / 2)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=ep).minimize(trainingloss)

            print("---------MSE Tensorflow---------")
            print("Batch Size: {}".format(batch_size))

        elif loss_function == "CE":
            train_y_hat = tf.sigmoid(tf.matmul(train_x, W) + b)
            loss = tf.losses.sigmoid_cross_entropy(train_y, train_y_hat)
            reg = tf.nn.l2_loss(W)
            trainingloss = loss + (beta_ce * reg / 2)

            valid_y_hat = tf.sigmoid(tf.matmul(valid_x, W) + b)
            loss = tf.losses.sigmoid_cross_entropy(valid_y, valid_y_hat)
            reg = tf.nn.l2_loss(W)
            validationloss = loss + (beta_ce * reg / 2)

            test_y_hat = tf.sigmoid(tf.matmul(test_x, W) + b)
            loss = tf.losses.sigmoid_cross_entropy(test_y, test_y_hat)
            reg = tf.nn.l2_loss(W)
            testingloss = loss + (beta_ce * reg / 2)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=ep).minimize(trainingloss)

            print("---------CE Tensorflow---------")
            print("Batch Size: {}".format(batch_size))

        with tf.Session(graph=g) as session:
            batch_num = int(3500 / batch_size)
            tf.global_variables_initializer().run()

            trainacc = []
            trainloss = []
            validacc = []
            validloss = []
            testacc = []
            testloss = []

            train_data, valid_data, test_data, train_label, valid_label, test_label = loadData()
            train_data = train_data.reshape(len(train_label), 28 * 28)
            valid_data = valid_data.reshape(len(valid_label), 28 * 28)
            test_data = test_data.reshape(len(test_label), 28 * 28)

            orig_time = time.time()

            for i in range(0, epochs):
                for j in range(0, batch_num):
                    x = train_data[j * batch_size : (j + 1) * batch_size, ]
                    y = train_label[j * batch_size : (j + 1) * batch_size, ]

                    _, new_W, new_b, tr_loss, tr_pred, v_loss, v_pred, te_loss, te_pred = session.run(
                        [optimizer, W, b, trainingloss, train_y_hat, validationloss, valid_y_hat,
                         testingloss, test_y_hat],
                        {train_x: x, train_y: y,
                         valid_x: valid_data, valid_y: valid_label,
                         test_x: test_data, test_y: test_label}
                    )

                trainloss.append(tr_loss)
                validloss.append(v_loss)
                testloss.append(te_loss)

                trainacc.append(accuracy_tf(tr_pred, y))
                validacc.append(accuracy_tf(v_pred, valid_label))
                testacc.append(accuracy_tf(te_pred, test_label))

                if i % 100 == 0:
                    runtime = time.time() - orig_time
                    print("Epoch: {}  Time Elapsed: {b:.3f}s".format(i, b=runtime))
                    print("Training Accuracy: {a:.5f}  Validation Accuracy: {b:.5f}".format(a=trainacc[-1], b=validacc[-1]))
                    print("Training Loss: {a:.5f}  Validation Loss: {b:.5f}".format(a=trainloss[-1], b=validloss[-1]))
                    print(" ")

    runtime = time.time() - orig_time

    print("-----Finished Training-----")
    print(" ")
    print("Epoch: {}  Time Elapsed: {b:.3f}s".format(epochs, b=runtime))
    print("Training Accuracy: {a:.5f}  Validation Accuracy: {b:.5f}".format(a=trainacc[-1], b=validacc[-1]))
    print("Training Loss: {a:.5f}  Validation Loss: {b:.5f}".format(a=trainloss[-1], b=validloss[-1]))
    print(" ")
    print("Testing Data Set")
    print("Testing Accuracy: {a:.5f}  Testing Loss: {b:.5f}".format(a=testacc[-1], b=testloss[-1]))
    print(" ")

    return new_W, new_b


# Other code

def accuracy(W, b, x, y):
    counter = 0
    y_hat = np.matmul(x, W) + b
    for i in range(0, len(y)):
        if y_hat[i][0] >= 0.5 and y[i][0] == 1:
            counter += 1
        elif y_hat[i][0] < 0.5 and y[i][0] == 0:
            counter += 1
    acc = counter / len(y)
    return acc


def accuracy_tf(y_hat, y):
    acc = (np.sum((y_hat >= 0.5) == y))
    acc = acc / len(y)
    return acc


def plot(training, validation, test, epochnum, is_accuracy):
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


# Main (Set arguments)


def main():
    # Random seed for numpy
    np.random.seed(421)

    # Initialize parameters and hyper-parameters
    # Parts 1 and 2
    W = np.random.rand(28*28, 1)
    b = 0
    alpha = 0.005
    epochs = 5000
    reg = 0.1
    error_tol = 1e-7
    epsilon = 1e-5

    # Part 3
    batch_size = 500
    lr = 0.001
    ep = 1e-5
    beta1 = 0
    beta2 = 0
    epochs_tf = 1500
    loss_function = "MSE"

    # Loading data
    train_x, valid_x, test_x, train_y, valid_y, test_y = loadData()

    # Part 1 MSELoss
    MSE_W, MSE_b, MSE_train, MSE_valid, MSE_test, MSE_epochnum = grad_descent(W, b, train_x, train_y, valid_x, valid_y,
                                                                              test_x, test_y, alpha, epochs, reg, error_tol)

    # Part 2 Cross Entropy Loss
    CE_W, CE_b, CE_train, CE_valid, CE_test, CE_epochnum = grad_descent_CE(W, b, train_x, train_y, valid_x, valid_y,
                                                                           test_x, test_y, alpha, epochs, reg, error_tol, epsilon)

    # Part 3 Tensorflow (MSE, CE)
    TF_W, TF_b = buildGraph(loss_function, batch_size, lr, ep, beta1, beta2, epochs_tf)

    # Plotting
    # This is for part 2.3 
    # epochforplot= MSE_epochnum
    # MSEtrainloss = MSE_train
    # CEtrainloss = CE_train
    # plt.plot(epochforplot, MSEtrainloss, label = 'MSE training loss')
    # plt.plot(epochforplot, CEtrainloss, label = 'CE trainig loss')
    # plt.title('MSEloss and CEloss vs. Number of Epoch')
    # plt.xlabel("Number of Epoch")
    # plt.ylabel("loss")
    # plt.legend(('MSEloss', 'CEloss'))
    # plt.show()


main()
