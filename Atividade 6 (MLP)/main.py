import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from training_test import TrainnigTestFolds
import os

def getPCA(fileToPCA, fileFromPCA, limiar):

    #Delete the Old File
    if os.path.exists(fileFromPCA):
        os.remove(fileFromPCA)
    #Open
    toBeProcess = pd.read_csv(fileToPCA)
    classes = toBeProcess['class']

    #Normalize
    scaler = StandardScaler().fit(toBeProcess)
    std = scaler.transform(toBeProcess)

    #PCA
    pca = PCA(n_components=12).fit(std)

    index = 1
    total = 0
    for component in pca.explained_variance_ratio_:
        if total > limiar:
            break
        index += 1
        total += component

    pca = PCA(n_components=index).fit(std)
    processed = pca.transform(std)

    #Save to csv
    data = pd.DataFrame(processed)
    csv = pd.concat([data, classes], axis=1)
    csv.to_csv(fileFromPCA)

# Create model
def main(num_steps, learning_rate, num_neuron, function, fileName, kPartes, ttf, num_hidden):
    # inicializa totais das estimativas
    tot_precision = 0
    tot_recall = 0
    tot_accuracy = 0

    file = open(fileName, "+a")
    file.write("\nTrainning Times: {} Hidden Layers: {} Neurons {} Learning Rate: {} \n".format(num_steps,
                                                                                              num_hidden,
                                                                                              num_neuron,
                                                                                              learning_rate))

    for i in range(0, kPartes - 1):

        df_training = ttf.k_df_training(i)
        df_test = ttf.k_df_test(i)
        target_training = pd.get_dummies(df_training['class'])
        target_test = pd.get_dummies(df_test['class'])

        # separa os atributos e as classes de teste e treinamento
        x_train = df_training.drop('class', axis=1)
        y_train = target_training
        x_test = df_test.drop('class', axis=1)
        y_test = target_test

        # Network Parameters
        num_input = x_train.shape[1] # data input
        num_classes = y_train.shape[1] # total classes

        # tf Graph input
        X = tf.placeholder("float", [None, num_input])
        Y = tf.placeholder("float", [None, num_classes])

        # Construct model
        logits = neural_net(X, num_neuron, function, num_input, num_classes)
        prediction = function(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=Y))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run([tf.global_variables_initializer()])

            display_step = 100
            for step in range(1, num_steps+1):
                batch_x, batch_y = x_train, y_train
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.4f}".format(acc))

            print("Optimization Finished!")

            TP = tf.count_nonzero(prediction * Y)
            TN = tf.count_nonzero((prediction - 1) * (Y - 1))
            FP = tf.count_nonzero(prediction * (Y - 1))
            FN = tf.count_nonzero((prediction - 1) * Y)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            acc, rec, pre = sess.run([accuracy, precision, recall], feed_dict={X: x_test, Y: y_test})

            tot_precision += (pre/kPartes)
            tot_recall += (rec/kPartes)
            tot_accuracy += (acc/kPartes)

    print("Testing Accuracy: {:.4f} Recall: {:.4f} Precision: {:.4f}\n\n\n".format(tot_accuracy,
                                                                                    tot_recall,
                                                                                    tot_precision));
    file.write("Testing Accuracy: {:.4f} Recall: {:.4f} Precision: {:.4f}\n\n\n".format(tot_accuracy,
                                                                                        tot_recall,
                                                                                        tot_precision))
    file.close()


def neural_net(x, num_neuron, function, num_input, num_classes):

    n_hidden_1 = num_neuron # 1st layer number of neurons
    #n_hidden_2 = num_neuron # 2nd layer number of neurons
    #n_hidden_3 = 20

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        #'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Hidden fully connected layer
    layer_1 = function(tf.matmul(x, weights['h1']) + biases['b1'])
    # Hidden fully connected layer
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


if __name__ == "__main__":
    # kpartes
    kPartes = 17


    # Parameters for Optimazer
    learning_rate = 0.01
    num_steps = 1500

    #Parameters for Network
    num_hidden = 1 #This is just to print 1 (To add new hidden layers. It has to be done mannualy)
    num_neuron = 200
    function = tf.nn.sigmoid

    # File
    fileFromDummies = "Data/dataset_dummies.csv"
    fileFromPCA = "Data/dataset_dummies_pca.csv"
    fileFromSelection = "Data/dataset_selection.csv"
    #Generate PCA
    getPCA(fileFromDummies, fileFromPCA, 0.75)

    fileToSaveResult = "Result/ResultProcessing.txt"

    #Para testar, apenas descomente um ()
    #With PCA
    #ttf = TrainnigTestFolds(fileFromPCA)

    #With Dummies
    #ttf = TrainnigTestFolds(fileFromDummies)

    #With Selection
    #ttf = TrainnigTestFolds(fileFromSelection)

    main(num_steps, learning_rate, num_neuron, function, fileToSaveResult, kPartes, ttf, num_hidden)
