import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imagemanagement as imagemanager
import imagedisplay as imagedisplayer
import globalsVar as gl

def get_mean(data, name, ifPrint = False):
    average = np.mean(data)
    if ifPrint: print("Average {}: {}".format(name, average))
    return average
    
def get():
    pretraining_data_path = imagemanager.get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    realImageNameTage = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    assert realImageNameTage, "Configure file error"
    AIImageNameTage = gl.App.get_config(section="IMAGE", option="AIImageNameTage", fallback=None)
    assert AIImageNameTage, "Configure file error"
    
    
    real_image_paths = ["apple-fruit-food-healthy.jpg", "apple-1539589_1280.jpg", "cherry-cox-apple.jpg", "apple-1539589_1280.jpg" ]
    ai_image_paths = ["_47cd24f2-afb7-4abf-a762-0a09d6428c29.jpg", "final (1).jpg", "image_fx_apple_in_a_tree_photography_realistic (1).jpg", "image_fx_apple_in_a_tree_photography_realistic (3).jpg"]
    
    real_image_paths = [ os.path.join(pretraining_data_path, realImageNameTage, f) for f in real_image_paths]
    ai_image_paths = [ os.path.join(pretraining_data_path, AIImageNameTage, f) for f in ai_image_paths]
    
    imagedisplayer.plot_images(real_image_paths, ai_image_paths)
    
    
    for i, _ in enumerate(real_image_paths):
        real_image = cv2.imread(real_image_paths[i])
        ai_image = cv2.imread(ai_image_paths[i])
        imagedisplayer.plot_image_histogram(real_image, ai_image)
        print()
    

def get_ela():
    pretraining_data_path = imagemanager.get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    realImageNameTage = gl.App.get_config(section="IMAGE", option="realImageNameTage", fallback=None)
    assert realImageNameTage, "Configure file error"
    AIImageNameTage = gl.App.get_config(section="IMAGE", option="AIImageNameTage", fallback=None)
    assert AIImageNameTage, "Configure file error"
    
    
    real_image_paths = ["apple-fruit-food-healthy_ela.jpg", "apple-1539589_1280_ela.jpg", "cherry-cox-apple_ela.jpg", "apple-1539589_1280_ela.jpg" ]
    ai_image_paths = ["_47cd24f2-afb7-4abf-a762-0a09d6428c29_ela.jpg", "final (1)_ela.jpg", "image_fx_apple_in_a_tree_photography_realistic (1)_ela.jpg", "image_fx_apple_in_a_tree_photography_realistic (3)_ela.jpg"]
    
    real_image_paths = [ os.path.join(pretraining_data_path, realImageNameTage, f) for f in real_image_paths]
    ai_image_paths = [ os.path.join(pretraining_data_path, AIImageNameTage, f) for f in ai_image_paths]
    
    imagedisplayer.plot_images(real_image_paths, ai_image_paths)
    
    
    for i, _ in enumerate(real_image_paths):
        real_image = cv2.imread(real_image_paths[i])
        ai_image = cv2.imread(ai_image_paths[i])
        imagedisplayer.plot_image_histogram(real_image, ai_image)
        print()

    
    

def _test1():
    get()


def main():
    get()
    
def main_ela():
    get_ela()
       
       
if __name__ == '__main__':
    # _test()
    _test1()