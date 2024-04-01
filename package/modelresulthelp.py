import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import imagemanagement as imagemanager
import globalsVar as gl
import network_input as network_inputer
import tensorflow as tf


from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import pandas as pd

def get_train_val_test_imgs_labels():
    data_path = imagemanager.get_data_path('IMAGE', 'dataPath', None, '..')
    assert data_path, "Configure file error"
    train_config = gl.App.get_config(section='IMAGE', option="train", fallback=None)
    assert train_config, "Configure file error"
    validation_config = gl.App.get_config(section='IMAGE', option="validation", fallback=None)
    assert validation_config, "Configure file error"
    test_config = gl.App.get_config(section='IMAGE', option="test", fallback=None)
    assert validation_config, "Configure file error"
    train_folder_path = os.path.join(data_path, train_config)
    val_folder_path = os.path.join(data_path, validation_config)
    test_folder_path = os.path.join(data_path, test_config)

    train_path = train_folder_path
    df_train = network_inputer.make_dataframe(train_path)
    train_images, train_labels = network_inputer.load_images(df_train, False)

    test_path = test_folder_path
    df_test = network_inputer.make_dataframe(test_path)
    test_images, test_labels = network_inputer.load_images(df_test, False)
    
    val_path = val_folder_path
    df_val = network_inputer.make_dataframe(val_path)
    val_images, val_labels = network_inputer.load_images(df_val, False)
    
    return [[train_images, train_labels], [val_images, val_labels], [test_images, test_labels]]

def normolization(images):
    return images / 255

def to_tensor(data):
    return tf.cast(data, tf.float32)


def unbatch_dataset(dataset):
    images = []
    labels = []

    # Iterate over the elements of the unbatched dataset
    for image, label in dataset.unbatch():
        images.append(image.numpy())
        labels.append(label.numpy())
    # labels = convert_onehot_lables_to_indices(labels)
    return {"images": np.array(images), "labels": labels}


def convert_onehot_lables_to_indices(onehot_labels):
    result = np.argmax(onehot_labels, axis=1) 
    return result

def predict_accuracy_classification(predicted_labels , true_labels):
    result = np.mean(predicted_labels == true_labels)
    return result

 
def show_prediction_result_classification(y_pred, y):
    print(classification_report(y, y_pred))
    
def plot_confusion_matrix(y_pred, y):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_pred)), annot=True, fmt='d', cmap='YlGnBu', alpha=0.8, vmin=0)
    plt.show()

def save_history(history, fn):
    with open(fn, 'wb') as fw:
        pickle.dump(history.history, fw, protocol=2)

def load_history(fn):
    class Temp():
        pass
    history = Temp()
    with open(fn, 'rb') as fr:
        history.history = pickle.load(fr)
    return history   


def get_ai_images_indexs(lables):
    realImageNameTage = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    assert realImageNameTage, "Configure file error"
    # using naive method to find indices for 3
    res_list = []
    for i in range(0, len(lables)):
        if lables[i] != realImageNameTage:
            res_list.append(i)
    return res_list

def get_ai_images_with_indexes(data, indexes):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return data[indexes]
    
    
    

def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.title('train Loss: %.3f' % history.history['loss'][-1])
    plt.plot(history.history["loss"], label="train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.legend() 
    plt.show()
    
    
    
def plot_encode_decode_images(n, normal_test_data, test_labels, decoded_imgs):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(normal_test_data[i])
        plt.title("original " + str(test_labels[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        # plt.title("reconstructed " + str(test_labels[i]))
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    
def plot_autoencoder_loss(test_loss_np_flat, title, color = "blue"):
    plt.hist(test_loss_np_flat, bins=50, color=color)
    plt.xlabel(title)
    plt.title("Distribution of " + title)
    plt.show()

def plot_autoencoder_loss_distributions(trainloss_flat, validationloss_flat, threshold):
    plt.hist(trainloss_flat, bins=50, label= "real")
    plt.hist(validationloss_flat, bins=50, label= "ai")
    plt.axvline(threshold, color = 'r', linewidth = 3, linestyle='dashed', label='{:0.3f}'.format(threshold))
    plt.legend(loc='upper right')
    plt.title("Distribution of Losses")
    plt.show()

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            previous = smoothed[-1]
            smoothed.append(previous * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

def plot_compare(history, steps=-1):
    if steps < 0:
        steps = len(history.history['accuracy'])
    accuracy = smooth_curve(history.history['accuracy'][:steps])
    val_acc = smooth_curve(history.history['val_accuracy'][:steps])
    loss = smooth_curve(history.history['loss'][:steps])
    val_loss = smooth_curve(history.history['val_loss'][:steps])
    
    plt.figure(figsize=(6, 4))
    plt.plot(loss, c='#0c7cba', label='Train Loss')
    plt.plot(val_loss, c='#0f9d58', label='Val Loss')
    plt.xticks(range(0, len(loss), 5))
    plt.xlim(0, len(loss))
    plt.title('Train Loss: %.3f, Val Loss: %.3f' % (loss[-1], val_loss[-1]), fontsize=12)
    plt.legend()
    
    plt.figure(figsize=(6, 4))
    plt.plot(accuracy, c='#0c7cba', label='Train accuracy')
    plt.plot(val_acc, c='#0f9d58', label='Val accuracy')
    plt.xticks(range(0, len(accuracy), 5))
    plt.xlim(0, len(accuracy))
    plt.title('Train Accuracy: %.3f, Val Accuracy: %.3f' % (accuracy[-1], val_acc[-1]), fontsize=12)
    plt.legend() 
    plt.show()