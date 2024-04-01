
import os

import numpy as np
import globalsVar as gl
import modelresulthelp as modelresulthelper
import network_input as network_inputer
import imagemanagement as imagemanager
import imagepreprocessing as imagepreprocessinger

from cnntf import CNN

import tensorflow as tf
from keras.optimizers import Adam
from sklearn.calibration import LabelEncoder
from matplotlib import pyplot as plt
from datetime import datetime

tf.random.set_seed(30)
np.random.seed(30)

def main():

    pretraining_data_path = imagemanager.get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    dateFormat_config = gl.App.get_config(section="DATAFORMAT", option="date_format", fallback=None)
    assert dateFormat_config, "Configure file error"
    modelresult_path = imagemanager.get_data_path('MODELRESULT', 'tensorflowpath', None, '..')
    assert modelresult_path, "Configure file error"
    
    
    date_obj = datetime.strptime("2024-03-05", dateFormat_config)
    images_path_full_info = imagemanager.get_image_files_path_full_info_from_raw_data_folder()
    
    # imagemanager.remove_all_files(pretraining_data_path)
    # imagemanager.move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, True, True, date_obj)
    imagemanager.remove_all_files(pretraining_data_path)
    imagemanager.move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)
    
    # # ela
    # imagemanager.remove_all_files(pretraining_data_path)
    # imagemanager.move_raw_image_ela_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)
    
    pretrain_images_path_full_info = imagemanager.get_image_with_x_classes_files_path_full_info_from_pretrain_folder()
    imagepreprocessinger.move_raw_image_to_des_folder_imageclassification(pretrain_images_path_full_info)

    n = 5
    batch_size = 32
    class_names = network_inputer.get_class_names()

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    ifShuffle = False



    get_train_val_test_imgs_labels_result = modelresulthelper.get_train_val_test_imgs_labels()
    train_data = get_train_val_test_imgs_labels_result[0][0]
    train_labels = get_train_val_test_imgs_labels_result[0][1]
    val_data = get_train_val_test_imgs_labels_result[1][0]
    val_labels = get_train_val_test_imgs_labels_result[1][1]
    test_data = get_train_val_test_imgs_labels_result[2][0]
    test_labels = get_train_val_test_imgs_labels_result[2][1]



    normal_train_data = modelresulthelper.normolization(train_data)
    normal_val_data = modelresulthelper.normolization(val_data)
    normal_test_data = modelresulthelper.normolization(test_data)

    normal_train_data = modelresulthelper.to_tensor(normal_train_data)
    normal_val_data = modelresulthelper.to_tensor(normal_val_data)
    normal_test_data = modelresulthelper.to_tensor(normal_test_data)



    train_labels_encoded = label_encoder.fit_transform(train_labels)
    # train_labels_onehot = tf.keras.utils.to_categorical(train_labels_encoded)
    train_dataset = tf.data.Dataset.from_tensor_slices((normal_train_data, train_labels_encoded))
    train_dataset = train_dataset.batch(batch_size)
    # train_dataset = train_dataset.shuffle(buffer_size=len(normal_train_data)).batch(batch_size)


    val_labels_encoded = label_encoder.fit_transform(val_labels)
    # val_labels_onehot = tf.keras.utils.to_categorical(val_labels_encoded)
    val_dataset = tf.data.Dataset.from_tensor_slices((normal_val_data, val_labels_encoded))
    val_dataset = val_dataset.batch(batch_size)
    # val_dataset = val_dataset.shuffle(buffer_size=len(normal_val_data)).batch(batch_size)


    train_dataset_unbatch = modelresulthelper.unbatch_dataset(train_dataset)

    # plt.imshow(train_dataset_unbatch["images"][0])
    # plt.show()
    # print(label_encoder.inverse_transform(train_dataset_unbatch["labels"]))



    learning_rate = 0.001
    imageClassifier = CNN(num_classes=len(class_names))
    imageClassifier.build((None, train_data[0].shape[0], train_data[0].shape[1], train_data[0].shape[2]))
    imageClassifier.summary()
    imageClassifier.compile(optimizer= Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])    

    epochs = 100  # Choose the number of epochs for training
   

    history = imageClassifier.fit(
                    train_dataset,
                    epochs=epochs,
                    shuffle=ifShuffle,
                    batch_size = batch_size,
                    validation_data=val_dataset)
    
    model_des_path = os.path.join(modelresult_path, 'cnn_saved_model')
    imageClassifier.save(model_des_path)
    
    model_hist_des_path = os.path.join(modelresult_path, 'cnn.bin')
    modelresulthelper.save_history(history, model_hist_des_path)

    modelresulthelper.plot_compare(history)
    # modelresulthelper.plot_loss(history)


    predictions = imageClassifier.predict(normal_test_data, verbose=0)
    predicted_labels = tf.argmax(predictions, axis=1)
    predicted_labels_inverse = label_encoder.inverse_transform(predicted_labels)
    # print(test_labels)
    # print(predicted_labels)
    # print(predicted_labels_inverse)


    true_labels_encoded = label_encoder.transform(test_labels)
    predicted_labels_inverse_encoded = label_encoder.transform(predicted_labels_inverse)
    # print(true_labels_encoded)
    # print(predicted_labels_inverse_encoded)

    modelresulthelper.show_prediction_result_classification(predicted_labels_inverse_encoded, true_labels_encoded)

    predict_accuracy_classification = modelresulthelper.predict_accuracy_classification(predicted_labels_inverse_encoded, true_labels_encoded)
    print(predict_accuracy_classification)


    modelresulthelper.plot_confusion_matrix(predicted_labels_inverse_encoded, true_labels_encoded)
    
    # history = modelresulthelper.load_history(model_hist_des_path)
    # modelresulthelper.plot_compare(history)


def main_ela():
    
    pretraining_data_path = imagemanager.get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    dateFormat_config = gl.App.get_config(section="DATAFORMAT", option="date_format", fallback=None)
    assert dateFormat_config, "Configure file error"
    modelresult_path = imagemanager.get_data_path('MODELRESULT', 'tensorflowpath', None, '..')
    assert modelresult_path, "Configure file error"
    
    
    date_obj = datetime.strptime("2024-03-05", dateFormat_config)
    images_path_full_info = imagemanager.get_image_files_path_full_info_from_raw_data_folder()
    
    # imagemanager.remove_all_files(pretraining_data_path)
    # imagemanager.move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, True, True, date_obj)
    # imagemanager.remove_all_files(pretraining_data_path)
    # imagemanager.move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)
    
    # ela
    imagemanager.remove_all_files(pretraining_data_path)
    imagemanager.move_raw_image_ela_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)
    
    pretrain_images_path_full_info = imagemanager.get_image_with_x_classes_files_path_full_info_from_pretrain_folder()
    imagepreprocessinger.move_raw_image_to_des_folder_imageclassification(pretrain_images_path_full_info)

    n = 5
    batch_size = 32
    class_names = network_inputer.get_class_names()

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    ifShuffle = True



    get_train_val_test_imgs_labels_result = modelresulthelper.get_train_val_test_imgs_labels()
    train_data = get_train_val_test_imgs_labels_result[0][0]
    train_labels = get_train_val_test_imgs_labels_result[0][1]
    val_data = get_train_val_test_imgs_labels_result[1][0]
    val_labels = get_train_val_test_imgs_labels_result[1][1]
    test_data = get_train_val_test_imgs_labels_result[2][0]
    test_labels = get_train_val_test_imgs_labels_result[2][1]



    normal_train_data = modelresulthelper.normolization(train_data)
    normal_val_data = modelresulthelper.normolization(val_data)
    normal_test_data = modelresulthelper.normolization(test_data)

    normal_train_data = modelresulthelper.to_tensor(normal_train_data)
    normal_val_data = modelresulthelper.to_tensor(normal_val_data)
    normal_test_data = modelresulthelper.to_tensor(normal_test_data)



    train_labels_encoded = label_encoder.fit_transform(train_labels)
    # train_labels_onehot = tf.keras.utils.to_categorical(train_labels_encoded)
    train_dataset = tf.data.Dataset.from_tensor_slices((normal_train_data, train_labels_encoded))
    train_dataset = train_dataset.batch(batch_size)
    # train_dataset = train_dataset.shuffle(buffer_size=len(normal_train_data)).batch(batch_size)


    val_labels_encoded = label_encoder.fit_transform(val_labels)
    # val_labels_onehot = tf.keras.utils.to_categorical(val_labels_encoded)
    val_dataset = tf.data.Dataset.from_tensor_slices((normal_val_data, val_labels_encoded))
    val_dataset = val_dataset.batch(batch_size)
    # val_dataset = val_dataset.shuffle(buffer_size=len(normal_val_data)).batch(batch_size)


    train_dataset_unbatch = modelresulthelper.unbatch_dataset(train_dataset)

    # plt.imshow(train_dataset_unbatch["images"][0])
    # plt.show()
    # print(label_encoder.inverse_transform(train_dataset_unbatch["labels"]))



    learning_rate = 0.001
    imageClassifier = CNN(num_classes=len(class_names))
    imageClassifier.build((None, train_data[0].shape[0], train_data[0].shape[1], train_data[0].shape[2]))
    imageClassifier.summary()
    imageClassifier.compile(optimizer= Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])    

    epochs = 100  # Choose the number of epochs for training
   

    history = imageClassifier.fit(
                    train_dataset,
                    epochs=epochs,
                    shuffle=ifShuffle,
                    batch_size = batch_size,
                    validation_data=val_dataset)
    
    model_des_path = os.path.join(modelresult_path, 'cnn_saved_model_ela')
    imageClassifier.save(model_des_path)
    
    model_hist_des_path = os.path.join(modelresult_path, 'cnn_ela.bin')
    modelresulthelper.save_history(history, model_hist_des_path)

    modelresulthelper.plot_compare(history)
    # modelresulthelper.plot_loss(history)


    predictions = imageClassifier.predict(normal_test_data, verbose=0)
    predicted_labels = tf.argmax(predictions, axis=1)
    predicted_labels_inverse = label_encoder.inverse_transform(predicted_labels)
    # print(test_labels)
    # print(predicted_labels)
    # print(predicted_labels_inverse)


    true_labels_encoded = label_encoder.transform(test_labels)
    predicted_labels_inverse_encoded = label_encoder.transform(predicted_labels_inverse)
    # print(true_labels_encoded)
    # print(predicted_labels_inverse_encoded)

    modelresulthelper.show_prediction_result_classification(predicted_labels_inverse_encoded, true_labels_encoded)

    predict_accuracy_classification = modelresulthelper.predict_accuracy_classification(predicted_labels_inverse_encoded, true_labels_encoded)
    print(predict_accuracy_classification)


    modelresulthelper.plot_confusion_matrix(predicted_labels_inverse_encoded, true_labels_encoded)
    
    # history = modelresulthelper.load_history(model_hist_des_path)
    # modelresulthelper.plot_compare(history)

if __name__ == '__main__':
    main()
    # main_ela()
    pass