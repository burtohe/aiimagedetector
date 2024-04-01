
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import imagemanagement as imagemanager
import globalsVar as gl

resizeImageWidth_config = gl.App.get_config(section='IMAGE', option="resizeImageWidth", fallback=None)
assert resizeImageWidth_config, "Configure file error"
resizeImageWidth = int(resizeImageWidth_config)
resizeImageHeigh_config = gl.App.get_config(section='IMAGE', option="resizeImageHeigh", fallback=None)
assert resizeImageHeigh_config, "Configure file error"
resizeImageHeigh = int(resizeImageHeigh_config)

def get_class_names():
    labels = []
    data_path = imagemanager.get_data_path('IMAGE', 'dataPath', None, '..')
    assert data_path, "Configure file error"
    test_config = gl.App.get_config(section='IMAGE', option="test", fallback=None)
    assert test_config, "Configure file error"
    test_folder_path = os.path.join(data_path, test_config)
    # print(test_folder_path)  
    # Assuming each class is stored in a separate subdirectory of the data directory
    classes = os.listdir(test_folder_path)
    for class_label, class_name in enumerate(classes):
        labels.append(class_name)
    return labels


def make_dataframe(data_directory, ifShuffle = False):
    # Define the directory where your images are stored
    data_directory = data_directory
    # Initialize lists to store file paths and labels
    file_paths = []
    labels = []
    # Assuming each class is stored in a separate subdirectory of the data directory
    classes = os.listdir(data_directory)

    # Iterate through each class directory and collect file paths and labels
    for class_label, class_name in enumerate(classes):
        class_directory = os.path.join(data_directory, class_name)
        for file_name in os.listdir(class_directory):
            file_path = os.path.join(class_directory, file_name)
            file_paths.append(file_path)
            labels.append(class_name)

    # Create a DataFrame from the collected file paths and labels
    df = pd.DataFrame({'file_path': file_paths, 'label': labels})
    if ifShuffle:
        # Optionally, shuffle the DataFrame
        df = df.sample(frac=1).reset_index(drop=True)
    return df


# Define a function to load images one by one
def load_images(df, grascale = False, color_mode = "rgb", image_height = resizeImageHeigh , image_width = resizeImageWidth):
    images = []
    labels = []
    for index, row in df.iterrows():
        file_path = row['file_path']
        label = row['label']
        img = tf.keras.preprocessing.image.load_img(file_path, grascale, color_mode, target_size=(image_height, image_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

def flattened_images(images):
    # Convert the list of sample images to a NumPy array
    if not isinstance(images, np.ndarray):
        images = np.array(images)
    # Reshape the array to flatten each image
    flattened_images = images.reshape(images.shape[0], -1)
    return flattened_images


def _test1():
    data_path = imagemanager.get_data_path('IMAGE', 'dataPath', None, '..')
    assert data_path, "Configure file error"
    train_config = gl.App.get_config(section='IMAGE', option="train", fallback=None)
    assert train_config, "Configure file error"
    validation_config = gl.App.get_config(section='IMAGE', option="validation", fallback=None)
    assert validation_config, "Configure file error"
    test_config = gl.App.get_config(section='IMAGE', option="test", fallback=None)
    assert validation_config, "Configure file error"

    train_folder_path = os.path.join(data_path, train_config)
    # print(train_folder_path)    
    val_folder_path = os.path.join(data_path, validation_config)
    # print(val_folder_path)    
    test_folder_path = os.path.join(data_path, test_config)
    # print(test_folder_path)  
    
    train_path = train_folder_path
    df_train = make_dataframe(train_path)
    train_images, train_labels = load_images(df_train, True)
    
    val_path = val_folder_path
    df_val = make_dataframe(val_path)
    val_images, val_labels = load_images(df_val, True)
    
    test_path = test_folder_path
    df_test = make_dataframe(test_path)
    test_images, test_labels = load_images(df_test, True)
    print(train_images.shape)
    
    
    flattened_train_images = flattened_images(train_images)
    # print(train_images[0].shape)  
    get_class_names()
    
    
if __name__ == '__main__':
    _test1()
    pass
