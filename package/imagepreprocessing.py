import os
import imagemanagement as imagemanager
import globalsVar as gl
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

train_config = gl.App.get_config(section='IMAGE', option="train", fallback=None)
assert train_config, "Configure file error"
validation_config = gl.App.get_config(section='IMAGE', option="validation", fallback=None)
assert validation_config, "Configure file error"
test_config = gl.App.get_config(section='IMAGE', option="test", fallback=None)
assert validation_config, "Configure file error"


def split_data_train_val_test(data, test_size = 0.2, random_state = 42, shuffle= True, debug = False):
    result = {}
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, shuffle= shuffle)
    train_data, val_data = train_test_split(train_data, test_size=test_size, random_state=random_state, shuffle= shuffle)
    result[train_config] = train_data
    result[validation_config] = val_data
    result[test_config] = test_data
    if debug:
        print("Number of train images:", len(train_data))
        print("Number of validation images:", len(val_data))
        print("Number of test images:", len(test_data))
    return result

def split_data_val_test(data, test_size = 0.5, random_state = 42, shuffle= True, debug = False):
    result = {}
    val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, shuffle= shuffle)
    result[validation_config] = val_data
    result[test_config] = test_data
    if debug:
        print("Validation set size:", len(val_data))
        print("Test set size:", len(test_data))
    return result

def move_raw_image_to_des_folder_autoencoder(pretrain_images_path_full_info):
    splitingResult = {}
    splitingResult[train_config] = {}
    splitingResult[validation_config] = {}
    splitingResult[test_config] = {}
    realImageNameTage_config = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    
    for image_class in pretrain_images_path_full_info:
        # print(image_class)
        if(image_class == realImageNameTage_config):
           real_images_path_split = split_data_train_val_test(pretrain_images_path_full_info[image_class])
           splitingResult[train_config][image_class] = real_images_path_split[train_config]
           splitingResult[validation_config][image_class] = real_images_path_split[validation_config]
           splitingResult[test_config][image_class] = real_images_path_split[test_config]
        #    print(real_images_path_split)
        else:
           ai_images_path_split = split_data_val_test(pretrain_images_path_full_info[image_class]) 
           splitingResult[train_config][image_class] = []
           splitingResult[validation_config][image_class] = ai_images_path_split[validation_config]
           splitingResult[test_config][image_class] = ai_images_path_split[test_config]
                
    data_path = imagemanager.get_data_path('IMAGE', 'dataPath', None, '..')
    assert data_path, "Configure file error"
    imagemanager.remove_all_files(data_path)
    # print(data_path)
    
    for dataset_type in splitingResult:
        # print(dataset_type)
        for image_type in splitingResult[dataset_type]:
            # print(image_type)
            des_folder_new = os.path.join(data_path, dataset_type,  image_type)
            if not os.path.exists(des_folder_new):
                    print (des_folder_new, "not found" )
                    os.makedirs(des_folder_new)
            for image in splitingResult[dataset_type][image_type]:
                file_path = Path(image)
                name = file_path.stem
                extension = file_path.suffix
                des_folder_new = os.path.join(data_path, dataset_type, image_type)
                # create folder
                if not os.path.exists(des_folder_new):
                    print (des_folder_new, "not found" )
                    os.makedirs(des_folder_new)
                new_name = os.path.join(des_folder_new, name + extension)
                shutil.copy(file_path, new_name)

def move_raw_image_to_des_folder_imageclassification(pretrain_images_path_full_info):
    splitingResult = {}
    splitingResult[train_config] = {}
    splitingResult[validation_config] = {}
    splitingResult[test_config] = {}
    realImageNameTage_config = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    
    for image_class in pretrain_images_path_full_info:
        if(image_class == realImageNameTage_config):
           real_images_path_split = split_data_train_val_test(pretrain_images_path_full_info[image_class])
           splitingResult[train_config][image_class] = real_images_path_split[train_config]
           splitingResult[validation_config][image_class] = real_images_path_split[validation_config]
           splitingResult[test_config][image_class] = real_images_path_split[test_config]
        #    print(real_images_path_split)
        else:
           ai_images_path_split = split_data_train_val_test(pretrain_images_path_full_info[image_class]) 
           splitingResult[train_config][image_class] = ai_images_path_split[train_config]
           splitingResult[validation_config][image_class] = ai_images_path_split[validation_config]
           splitingResult[test_config][image_class] = ai_images_path_split[test_config]
                
    data_path = imagemanager.get_data_path('IMAGE', 'dataPath', None, '..')
    assert data_path, "Configure file error"
    imagemanager.remove_all_files(data_path)
    # print(data_path)
    
    for dataset_type in splitingResult:
        # print(dataset_type)
        for image_type in splitingResult[dataset_type]:
            # print(image_type)
            des_folder_new = os.path.join(data_path, dataset_type,  image_type)
            if not os.path.exists(des_folder_new):
                    print (des_folder_new, "not found" )
                    os.makedirs(des_folder_new)
            for image in splitingResult[dataset_type][image_type]:
                file_path = Path(image)
                name = file_path.stem
                extension = file_path.suffix
                des_folder_new = os.path.join(data_path, dataset_type, image_type)
                # create folder
                if not os.path.exists(des_folder_new):
                    print (des_folder_new, "not found" )
                    os.makedirs(des_folder_new)
                new_name = os.path.join(des_folder_new, name + extension)
                shutil.copy(file_path, new_name)
    
def _test1():
    pretrain_images_path_full_info = imagemanager.get_image_with_x_classes_files_path_full_info_from_pretrain_folder()
    move_raw_image_to_des_folder_autoencoder(pretrain_images_path_full_info)
    
    
    pretrain_images_path_full_info = imagemanager.get_image_with_x_classes_files_path_full_info_from_pretrain_folder()
    move_raw_image_to_des_folder_imageclassification(pretrain_images_path_full_info)


def main_autoencoder():
    pretrain_images_path_full_info = imagemanager.get_image_with_x_classes_files_path_full_info_from_pretrain_folder()
    move_raw_image_to_des_folder_autoencoder(pretrain_images_path_full_info)


def main_classification():
    pretrain_images_path_full_info = imagemanager.get_image_with_x_classes_files_path_full_info_from_pretrain_folder()
    move_raw_image_to_des_folder_imageclassification(pretrain_images_path_full_info)
 


if __name__ == '__main__':
    _test1()
    pass