# -*- coding: utf-8 -*-
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Get the list of available GPUs
gpus = tf.config.list_physical_devices('GPU')
# If a GPU is detected, set the first GPU as visible to the current program
if gpus:
  try:
    # Select the first GPU for training
    tf.config.set_visible_devices(gpus[0], 'GPU')
    # Set GPU memory growth to avoid allocating all memory at once
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Using GPU: ", gpus[0].name)
  except RuntimeError as e:
    # If an error occurs, print the error message
    print("Error setting up GPU:", e)
else:
  print("No GPU found. Using CPU instead.")

import numpy as np
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import DenseNet121 as md
import datetime
from HGAO import HGAO

np.set_printoptions(threshold=np.inf)

def create_train_data():
    from read_data import x_data,y_data

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    print(tf.shape(x_train))
    print(tf.shape(y_test))
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # Convert labeled data to int
    y_test = label_encoder.fit_transform(y_test)  # Convert labeled data to int
    return x_train, y_train, x_test, y_test

def model_predict(cnn, model_name, it_acc, it_val_acc, it_loss, it_val_loss, model_param,Read_time,best_X):

    test_data=model_param['test_data']
    test_label=model_param['test_label']
    score = cnn.evaluate(test_data, test_label, verbose=1)
    test_loss = 'Test Loss : {:.4f}'.format(score[0])
    test_accuracy = 'Test Accuracy : {:.4f}'.format(score[1])
    predicted_probabilities = cnn.predict(test_data)
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    correct = np.nonzero(predicted_classes == test_label)
    incorrect = np.nonzero(predicted_classes != test_label)
    acc_score = accuracy_score(test_label, predicted_classes)
    cls_report = classification_report(test_label, predicted_classes, zero_division=1)
    print(test_loss)
    print(test_accuracy)

    # Calculate the confusion matrix
    cm = confusion_matrix(test_label, predicted_classes)
    # Get the number of classes
    num_classes = len(np.unique(test_label))
    # Calculate the FPR and TPR for each class
    fpr = np.zeros(num_classes)
    tpr = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
        fn = np.sum(cm[i, :]) - tp
        print(tp, fp, tn, fn)

        fpr[i] = fp / (fp + tn)
        tpr[i] = tp / (tp + fn)

    # Save data to file
    current_time = datetime.datetime.now()
    time = current_time - Read_time
    sss = time.total_seconds()

    if not os.path.exists('result/' + version):
        os.mkdir('result/' + version)
    file = open("./result/" + version + '/record_' + model_name + '.txt', "a")
    file.write("The optimal hyperparameter is:" + str(best_X) + '\n')
    file.write(test_loss)
    file.write('\ntest_accuracy：' + test_accuracy)
    file.write('\ncorrect：' + str(len(correct[0])))
    file.write('\nincorrect：' + str(len(incorrect[0])))
    file.write('\nacc_score：' + str(acc_score))
    file.write('\n' + cls_report)

    # Print FPR and TPR for each category
    for i in range(num_classes):
        file.write(f"Class {i} - FPR: {fpr[i]}, TPR: {tpr[i]}" + '\n')

    file.write('\n' + "it_acc=" + str(it_acc))
    file.write('\n' + "it_val_acc=" + str(it_val_acc))
    file.write('\n' + "it_loss=" + str(it_loss))
    file.write('\n' + "it_val_loss=" + str(it_val_loss))
    file.write('\n' + "Time spent: " + str(sss))
    file.close()
    # Save the model
    if not os.path.exists('models/' + version):
        os.mkdir('models/' + version)
    model_path = 'models/' + version + '/model_' + model_name + '.h5'
    cnn.save(model_path)
    print("Model training completed, saved at:", model_path)
    print("Total time spent:", sss)

if __name__ == '__main__':
    # Generate training and testing datasets
    Read_time = datetime.datetime.now()
    train_data, train_label, test_data, test_label = create_train_data()
    # Model parameters
    model_param = {
        "test_data": test_data,
        "test_label": test_label,
        "data": train_data,
        "label": train_label
    }
    hgao_param = {
        "dim": 2,  # What are the dimensions of lb and ub? This is what
        "SearchAgents": 10,
        "Max_iter": 10,

        "lb": np.array([0.00001, 0.1]),  # lowbound
        "ub": np.array([0.1, 0.9])  # upbound
    }
    version = 'hgao'
    import os

    folder_path = 'result/' + version

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    folder_path = 'models/' + version
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")


    gao=HGAO(model_param, hgao_param,0.5,0.5)
    Optimal_loss, Optimal_hyperparameters = gao.run()

    print("Optimal validation loss after joint model optimization:", Optimal_loss)
    print("Optimal hyperparameters after joint model optimization:", Optimal_hyperparameters)


    print(tf.config.list_physical_devices())
    with tf.device('/device:GPU:0'):
        densenet_model = md.DenseNet121(Optimal_hyperparameters)
        hgao_model = densenet_model.model_create(Optimal_hyperparameters[0])
        hgao_history = hgao_model.fit(train_data, train_label, epochs=60, batch_size=16, validation_split=0.1)
    # Calculating evaluation metrics
    hgao_accuracy = hgao_history.history['accuracy']
    hgao_val_accuracy = hgao_history.history['val_accuracy']
    hgao_loss = hgao_history.history['loss']
    hgao_val_loss = hgao_history.history['val_loss']

    model_predict(hgao_model, version,
                      hgao_accuracy, hgao_val_accuracy, hgao_loss, hgao_val_loss, model_param,Read_time,Optimal_hyperparameters)
