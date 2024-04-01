import os

import numpy as np
import globalsVar as gl
import modelresulthelp as modelresulthelper
import network_input as network_inputer
import imagemanagement as imagemanager
import imagepreprocessing as imagepreprocessinger

from autoencodertf import Autoencoder

import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras import layers, losses

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
    imagepreprocessinger.move_raw_image_to_des_folder_autoencoder(pretrain_images_path_full_info)

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
    
    # using naive method to find indices for 3
    AIImageNameTage = gl.App.get_config(section="IMAGE", option="AIImageNameTage", fallback=None)
    assert AIImageNameTage, "Configure file error"
    ai_images_val_labels = [AIImageNameTage] * n
    val_ai_list = modelresulthelper.get_ai_images_indexs(val_labels)
    ai_images_val =modelresulthelper.get_ai_images_with_indexes(val_data, val_ai_list)
    
    normal_ai_images_val = modelresulthelper.normolization(ai_images_val)
    normal_ai_images_val = modelresulthelper.to_tensor(normal_ai_images_val)
    
    
    shape = normal_train_data.shape[1:]
    latent_dim = 8
    autoencoder = Autoencoder(latent_dim, shape)

    # autoencoder = AnomalyDetector()
    # autoencoder.build((None, shape[0], shape[1], shape[2]))
    # autoencoder.summary()
    autoencoder.summary_alt(latent_dim,shape)
    autoencoder.compile(optimizer= Adam(learning_rate=0.001), loss=losses.MeanSquaredError() , metrics=['accuracy'])

    epochs = 100  # Choose the number of epochs for training
    
    history = autoencoder.fit(normal_train_data, normal_train_data,
                    epochs=epochs,
                    shuffle=ifShuffle,
                    batch_size = batch_size,
                    validation_data=(normal_val_data, normal_val_data))
    
    model_des_path = os.path.join(modelresult_path, 'autoencoder_saved_model')
    autoencoder.save(model_des_path)
    
    model_hist_des_path = os.path.join(modelresult_path, 'autoencoderhistory.bin')
    modelresulthelper.save_history(history, model_hist_des_path)


    modelresulthelper.plot_compare(history)
    # modelresulthelper.plot_loss(history)
    
    
    reconstructions = autoencoder.predict(normal_train_data)
    
    modelresulthelper.plot_encode_decode_images(n, normal_train_data, train_labels, reconstructions)
    
    reconstructions = autoencoder.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
    train_loss_np = train_loss.numpy()
    # train_loss = np.mean(np.square(normal_train_data - reconstructions), axis=(1, 2, 3))
    # train_loss_np = train_loss
    train_loss_np_flat = train_loss_np.flatten()
    modelresulthelper.plot_autoencoder_loss(train_loss_np_flat, "Train Loss")
    
    threshold = np.mean(train_loss) + np.std(train_loss)
    # threshold = np.mean(train_loss)
    print("Threshold: ", threshold)
    
    
    
    reconstructions_val = autoencoder.predict(normal_val_data)
    val_loss = tf.keras.losses.mae(reconstructions_val, normal_val_data)
    val_loss_np = val_loss.numpy()
    # val_loss = np.mean(np.square(normal_val_data - reconstructions_val), axis=(1, 2, 3))
    # val_loss_np = val_loss
    
    val_loss_np_flat = val_loss_np.flatten()
    modelresulthelper.plot_autoencoder_loss(val_loss_np_flat, "Validation Loss")




    reconstructions_val_ai = autoencoder.predict(normal_ai_images_val)    
    
    modelresulthelper.plot_encode_decode_images(n, normal_ai_images_val, ai_images_val_labels, reconstructions_val_ai)
    
    
    val_loss_ai = tf.keras.losses.mae(reconstructions_val_ai, normal_ai_images_val)
    val_loss_ai_np = val_loss_ai.numpy()
    # val_loss_ai = np.mean(np.square(normal_ai_images_val - reconstructions_val_ai), axis=(1, 2, 3))
    # val_loss_ai_np = val_loss_ai
    val_loss_ai_np_flat = val_loss_ai_np.flatten()
    modelresulthelper.plot_autoencoder_loss(val_loss_ai_np_flat, "Validation ai Loss")

    modelresulthelper.plot_autoencoder_loss_distributions(train_loss_np_flat, val_loss_ai_np_flat, threshold)

    
    reconstructed_test = autoencoder.predict(normal_test_data)
    test_loss = np.mean(np.square(test_data - reconstructed_test), axis=(1, 2, 3))
    predictions = (test_loss > threshold).astype(int)
    
    true_labels_encoded = label_encoder.transform(test_labels)
    # print(test_labels)
    # print(true_labels_encoded)
    # print(predictions)
    
    predict_accuracy_classification = modelresulthelper.predict_accuracy_classification(predictions, true_labels_encoded)
    print(predict_accuracy_classification)
    
    modelresulthelper.plot_confusion_matrix(predictions, true_labels_encoded)
    

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
    imagepreprocessinger.move_raw_image_to_des_folder_autoencoder(pretrain_images_path_full_info)

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
    
    # using naive method to find indices for 3
    AIImageNameTage = gl.App.get_config(section="IMAGE", option="AIImageNameTage", fallback=None)
    assert AIImageNameTage, "Configure file error"
    ai_images_val_labels = [AIImageNameTage] * n
    val_ai_list = modelresulthelper.get_ai_images_indexs(val_labels)
    ai_images_val =modelresulthelper.get_ai_images_with_indexes(val_data, val_ai_list)
    
    normal_ai_images_val = modelresulthelper.normolization(ai_images_val)
    normal_ai_images_val = modelresulthelper.to_tensor(normal_ai_images_val)
    
    
    shape = normal_train_data.shape[1:]
    latent_dim = 8
    autoencoder = Autoencoder(latent_dim, shape)

    # autoencoder = AnomalyDetector()
    # autoencoder.build((None, shape[0], shape[1], shape[2]))
    # autoencoder.summary()
    autoencoder.summary_alt(latent_dim,shape)
    autoencoder.compile(optimizer= Adam(learning_rate=0.001), loss=losses.MeanSquaredError() , metrics=['accuracy'])

    epochs = 100  # Choose the number of epochs for training
    
    history = autoencoder.fit(normal_train_data, normal_train_data,
                    epochs=epochs,
                    shuffle=ifShuffle,
                    batch_size = batch_size,
                    validation_data=(normal_val_data, normal_val_data))
    
    model_des_path = os.path.join(modelresult_path, 'autoencoder_saved_model_ela')
    autoencoder.save(model_des_path)
    
    model_hist_des_path = os.path.join(modelresult_path, 'autoencoderhistory_ela.bin')
    modelresulthelper.save_history(history, model_hist_des_path)


    modelresulthelper.plot_compare(history)
    # modelresulthelper.plot_loss(history)
    
    
    reconstructions = autoencoder.predict(normal_train_data)
    
    modelresulthelper.plot_encode_decode_images(n, normal_train_data, train_labels, reconstructions)
    
    reconstructions = autoencoder.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
    train_loss_np = train_loss.numpy()
    # train_loss = np.mean(np.square(normal_train_data - reconstructions), axis=(1, 2, 3))
    # train_loss_np = train_loss
    train_loss_np_flat = train_loss_np.flatten()
    modelresulthelper.plot_autoencoder_loss(train_loss_np_flat, "Train Loss")
    
    threshold = np.mean(train_loss) + np.std(train_loss)
    # threshold = np.mean(train_loss)
    print("Threshold: ", threshold)
    
    
    
    reconstructions_val = autoencoder.predict(normal_val_data)
    val_loss = tf.keras.losses.mae(reconstructions_val, normal_val_data)
    val_loss_np = val_loss.numpy()
    # val_loss = np.mean(np.square(normal_val_data - reconstructions_val), axis=(1, 2, 3))
    # val_loss_np = val_loss
    
    val_loss_np_flat = val_loss_np.flatten()
    modelresulthelper.plot_autoencoder_loss(val_loss_np_flat, "Validation Loss")




    reconstructions_val_ai = autoencoder.predict(normal_ai_images_val)    
    
    modelresulthelper.plot_encode_decode_images(n, normal_ai_images_val, ai_images_val_labels, reconstructions_val_ai)
    
    
    val_loss_ai = tf.keras.losses.mae(reconstructions_val_ai, normal_ai_images_val)
    val_loss_ai_np = val_loss_ai.numpy()
    # val_loss_ai = np.mean(np.square(normal_ai_images_val - reconstructions_val_ai), axis=(1, 2, 3))
    # val_loss_ai_np = val_loss_ai
    val_loss_ai_np_flat = val_loss_ai_np.flatten()
    modelresulthelper.plot_autoencoder_loss(val_loss_ai_np_flat, "Validation ai Loss")

    modelresulthelper.plot_autoencoder_loss_distributions(train_loss_np_flat, val_loss_ai_np_flat, threshold)

    
    reconstructed_test = autoencoder.predict(normal_test_data)
    test_loss = np.mean(np.square(test_data - reconstructed_test), axis=(1, 2, 3))
    predictions = (test_loss > threshold).astype(int)
    
    true_labels_encoded = label_encoder.transform(test_labels)
    # print(test_labels)
    # print(true_labels_encoded)
    # print(predictions)
    
    predict_accuracy_classification = modelresulthelper.predict_accuracy_classification(predictions, true_labels_encoded)
    print(predict_accuracy_classification)
    
    modelresulthelper.plot_confusion_matrix(predictions, true_labels_encoded)

    
    

if __name__ == '__main__':
    main()
    # main_ela()
    pass