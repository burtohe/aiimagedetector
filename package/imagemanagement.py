#!/usr/bin/env python3


import os
import shutil

import cv2
import globalsVar as gl
import imagedisplay as imagedisplayer
import ela as ela_converter
from datetime import datetime
from PIL import Image
from pathlib import Path

def detect_image_format(file_path):
    try:
        with Image.open(file_path) as img:
            return img.format
    except IOError:
        return None

def get_data_path(section, option, fallback=None, parent='..'):
    dataPath_config = gl.App.get_config(section=section, option=option, fallback=fallback)
    assert dataPath_config, "Configure file error"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), parent , str(dataPath_config)))
    


def remove_unsupported_image_type_from_raw_data_folder():
    raw_data_path = get_data_path('IMAGE', 'rawDataPath', None, '..')
    
    image_exts = gl.App.get_config(section='IMAGE', option='acceptImageType', fallback=None).split("\n")
    for image_models in os.listdir(raw_data_path): 
        image_models_path = os.path.join(raw_data_path, image_models)
        for image_class_with_date in os.listdir(image_models_path):
            image_class_with_date_path = os.path.join(raw_data_path, image_models, image_class_with_date)
            for image_catergory in os.listdir(image_class_with_date_path):
                image_catergory_path = os.path.join(raw_data_path, image_models, image_class_with_date, image_catergory)
                if os.path.isdir(image_catergory_path):
                    for image in os.listdir(image_catergory_path):
                        image_path = os.path.join(raw_data_path, image_models, image_class_with_date, image_catergory, image)
                        try:
                            raw_s = r'{}'.format(image_path)
                            tip = detect_image_format(raw_s).lower()
                            if tip not in image_exts:
                                print("{} Image not in ext list {}, removing".format(tip, image_path))
                                os.remove(raw_s)
                        except Exception as e:
                            # assert False, "Issue with image {}".format(image_path)
                            print("Issue with image {}".format(image_path))
                            return False
    return True

def get_image_files_path_full_info_from_raw_data_folder():
    result_list = {}
    raw_data_path = get_data_path('IMAGE', 'rawDataPath', None, '..')
    for image_models in os.listdir(raw_data_path): 
        result_list[image_models] = {}
        image_models_path = os.path.join(raw_data_path, image_models)
        for image_class_with_date in os.listdir(image_models_path):
            
            result_list[image_models][image_class_with_date] = []
            image_class_with_date_path = os.path.join(raw_data_path, image_models, image_class_with_date)
            for image_catergory in os.listdir(image_class_with_date_path):
                image_catergory_path = os.path.join(raw_data_path, image_models, image_class_with_date, image_catergory)
                if os.path.isdir(image_catergory_path):
                    for image in os.listdir(image_catergory_path):
                        image_path = os.path.join(raw_data_path, image_models, image_class_with_date, image_catergory, image)
                        raw_s = r'{}'.format(image_path)
                        result_list[image_models][image_class_with_date].append(raw_s)
    return result_list

def get_image_files_path_only_from_raw_data_folder():
    result_list = []
    images_path = get_image_files_path_full_info_from_raw_data_folder()
    
    for k1, v1 in (images_path.items()):
        for k2, v2 in (v1.items()):
            for k3, v3 in enumerate(v2):
                result_list.append(v3)
    return result_list

def remove_all_files(folder):
    folder = folder
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def move_raw_image_to_des_folder(images_path_full_info, des_folder, ifModel = False, ifDate = False , dateCompare= datetime.now()):
    
    dateFormat_config = gl.App.get_config(section="DATAFORMAT", option="date_format", fallback=None)
    assert dateFormat_config, "Configure file error"
    
    realImageNameTage = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    assert realImageNameTage, "Configure file error"
    AIImageNameTage = gl.App.get_config(section="IMAGE", option="AIImageNameTage", fallback=None)
    assert AIImageNameTage, "Configure file error"
    
    for image_model in images_path_full_info:
        for image_class in images_path_full_info[image_model]:
            for image in images_path_full_info[image_model][image_class]:
                imageType, imageClass, imageDate = (image_class.split("_"))
                date_obj = datetime.strptime(imageDate, dateFormat_config)

                file_path = Path(image)
                name = file_path.stem
                extension = file_path.suffix
                
                if not ifModel:
                    # des_folder_new = os.path.join(des_folder, imageType)
                    # new_name = os.path.join(des_folder_new, name + extension)
                    if imageType == "ai":
                        des_folder_new = os.path.join(des_folder, AIImageNameTage)
                        new_name = os.path.join(des_folder_new, name + extension)
                    elif imageType == "real":
                        des_folder_new = os.path.join(des_folder, realImageNameTage )
                        new_name = os.path.join(des_folder_new, name + extension)
                    else:
                        assert False, "Image Either real or ai, path: {}".format(file_path)   
                else:
                    if imageType == "ai":
                        des_folder_new = os.path.join(des_folder, image_model)
                        new_name = os.path.join(des_folder_new, name + extension)
                    elif imageType == "real":
                        des_folder_new = os.path.join(des_folder, realImageNameTage )
                        new_name = os.path.join(des_folder_new, name + extension)
                    else:
                        assert False, "Image Either real or ai, path: {}".format(file_path)   
                        
                # create folder
                if not os.path.exists(des_folder_new):
                    print (des_folder_new, "not found" )
                    os.makedirs(des_folder_new)
                    
                if (not ifDate) or (ifDate and date_obj >= dateCompare):
                    
                    if not os.path.exists(new_name):  # folder exists, file does not
                        shutil.copy(file_path, new_name)
                    else:  # folder exists, file exists as well
                        ii = 1
                        while True:
                            new_name = os.path.join(des_folder_new, name + "_" + str(ii) + extension)
                            if not os.path.exists(new_name):
                                try:
                                    shutil.copy(file_path, new_name)
                                    print ("Copied", file_path, "as", new_name)
                                    break 
                                except:
                                    assert False, "copy file error, {}".format(file_path)
                            ii += 1

def move_raw_image_ela_to_des_folder(images_path_full_info, des_folder, ifModel = False, ifDate = False , dateCompare= datetime.now()):
    
    dateFormat_config = gl.App.get_config(section="DATAFORMAT", option="date_format", fallback=None)
    assert dateFormat_config, "Configure file error"
    
    realImageNameTage = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    assert realImageNameTage, "Configure file error"
    AIImageNameTage = gl.App.get_config(section="IMAGE", option="AIImageNameTage", fallback=None)
    assert AIImageNameTage, "Configure file error"
    
    ela_name_tag = "_ela"
    
    for image_model in images_path_full_info:
        for image_class in images_path_full_info[image_model]:
            for image in images_path_full_info[image_model][image_class]:
                imageType, imageClass, imageDate = (image_class.split("_"))
                date_obj = datetime.strptime(imageDate, dateFormat_config)

                file_path = Path(image)
                name = file_path.stem
                extension = file_path.suffix
                
                if not ifModel:
                    # des_folder_new = os.path.join(des_folder, imageType)
                    # new_name = os.path.join(des_folder_new, name + extension)
                    if imageType == "ai":
                        des_folder_new = os.path.join(des_folder, AIImageNameTage)
                        new_name = os.path.join(des_folder_new, name + ela_name_tag + extension)
                    elif imageType == "real":
                        des_folder_new = os.path.join(des_folder, realImageNameTage )
                        new_name = os.path.join(des_folder_new, name + ela_name_tag + extension)
                    else:
                        assert False, "Image Either real or ai, path: {}".format(file_path)   
                else:
                    if imageType == "ai":
                        des_folder_new = os.path.join(des_folder, image_model)
                        new_name = os.path.join(des_folder_new, name + ela_name_tag + extension)
                    elif imageType == "real":
                        des_folder_new = os.path.join(des_folder, realImageNameTage )
                        new_name = os.path.join(des_folder_new, name + ela_name_tag + extension)
                    else:
                        assert False, "Image Either real or ai, path: {}".format(file_path)   
                        
                # create folder
                if not os.path.exists(des_folder_new):
                    print (des_folder_new, "not found" )
                    os.makedirs(des_folder_new)
                
                file_path_str = str(file_path)
                image = cv2.imread(file_path_str)
                ela_result = ela_converter.ela(image, des_folder, quality = 95, scale = 15)
                # ela_result = cv2.imread("temp_ela.png")
                # cv2.imshow("ela95", ela_result)
                # cv2.waitKey(0)
                    
                if (not ifDate) or (ifDate and date_obj >= dateCompare):
                    
                    if not os.path.exists(new_name):  # folder exists, file does not
                        # shutil.copy(file_path, new_name)
                        cv2.imwrite(new_name, ela_result) 
                    else:  # folder exists, file exists as well
                        ii = 1
                        while True:
                            new_name = os.path.join(des_folder_new, name + "_" + str(ii) + ela_name_tag + extension)
                            if not os.path.exists(new_name):
                                try:
                                    # shutil.copy(file_path, new_name)
                                    cv2.imwrite(new_name, ela_result) 
                                    print ("Copied", file_path, "as", new_name)
                                    break 
                                except:
                                    assert False, "copy file error, {}".format(file_path)
                            ii += 1

def get_image_files_path_full_info_from_pretrain_folder():
    result_list = {}
    pretrain_data_path = get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    for image_class in os.listdir(pretrain_data_path): 
        result_list[image_class] = []
        image_path = os.path.join(pretrain_data_path, image_class)
        for image in os.listdir(image_path):
            image_path = os.path.join(pretrain_data_path, image_class, image)
            raw_s = r'{}'.format(image_path)
            result_list[image_class].append(raw_s)
    return result_list

def get_image_with_x_classes_files_path_full_info_from_pretrain_folder():
    result_list = {}
    # result_list["real"] = []
    
    pretrain_data_path = get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    for image_class in os.listdir(pretrain_data_path):
        image_class_info = image_class.split("_")
        image_path = os.path.join(pretrain_data_path, image_class)
        result_list[image_class] = []
        for image in os.listdir(image_path):
            image_path = os.path.join(pretrain_data_path, image_class, image)
            raw_s = r'{}'.format(image_path)
            # if (image_class_info[0] != "real"):
            #     # print(image_class_info)
            result_list[image_class].append(raw_s)
            # else:
            #     result_list["real"].append(raw_s)
    return result_list


def get_x_classes_from_pretrain_folder():
    result_list = []
    pretrain_data_path = get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    for image_class in os.listdir(pretrain_data_path):
        result_list.append(image_class)
    # place real_intent to the end
    realImageNameTage_config = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    result_list.remove(realImageNameTage_config)
    result_list.insert(len(result_list), realImageNameTage_config)
    return result_list

    
def _test1():
    remove_result = remove_unsupported_image_type_from_raw_data_folder()
    assert remove_result, "Issue with image removing"
    
    images_path = get_image_files_path_full_info_from_raw_data_folder()
    for image_model in images_path:
        for image_class in images_path[image_model]:
            for image in images_path[image_model][image_class]:
                imagedisplayer.show_image_with_path(image)
                print(image)

def _test2():
    
    pretraining_data_path = get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    dateFormat_config = gl.App.get_config(section="DATAFORMAT", option="date_format", fallback=None)
    assert dateFormat_config, "Configure file error"
    
    date_obj = datetime.strptime("2024-03-05", dateFormat_config)
    images_path_full_info = get_image_files_path_full_info_from_raw_data_folder()
    remove_all_files(pretraining_data_path)
    move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, True, True, date_obj)
    remove_all_files(pretraining_data_path)
    move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)
    
    # # ela
    # remove_all_files(pretraining_data_path)
    # move_raw_image_ela_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)


def main():
    pretraining_data_path = get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    dateFormat_config = gl.App.get_config(section="DATAFORMAT", option="date_format", fallback=None)
    assert dateFormat_config, "Configure file error"
    
    date_obj = datetime.strptime("2024-03-05", dateFormat_config)
    images_path_full_info = get_image_files_path_full_info_from_raw_data_folder()
    # remove_all_files(pretraining_data_path)
    # move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, True, True, date_obj)
    remove_all_files(pretraining_data_path)
    move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)


def main_ela():
    pretraining_data_path = get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    dateFormat_config = gl.App.get_config(section="DATAFORMAT", option="date_format", fallback=None)
    assert dateFormat_config, "Configure file error"
    
    date_obj = datetime.strptime("2024-03-05", dateFormat_config)
    images_path_full_info = get_image_files_path_full_info_from_raw_data_folder()
    # ela
    # remove_all_files(pretraining_data_path)
    # move_raw_image_to_des_folder(images_path_full_info, pretraining_data_path, True, True, date_obj)
    remove_all_files(pretraining_data_path)
    move_raw_image_ela_to_des_folder(images_path_full_info, pretraining_data_path, False, False, date_obj)
       
if __name__ == '__main__':
    # _test1()
    _test2()
    
    # print(get_image_files_path_full_info_from_pretrain_folder())
    
    # print(get_image_files_path_only_from_raw_data_folder())
    
    # print(get_image_with_x_classes_files_path_full_info_from_pretrain_folder())
    
    # print(get_x_classes_from_pretrain_folder())
    
    
    pass