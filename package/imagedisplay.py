import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import globalsVar as gl


resizeImageWidth_config = gl.App.get_config(section='IMAGE', option="resizeImageWidth", fallback=None)
assert resizeImageWidth_config, "Configure file error"
resizeImageWidth = int(resizeImageWidth_config)
resizeImageHeigh_config = gl.App.get_config(section='IMAGE', option="resizeImageHeigh", fallback=None)
assert resizeImageHeigh_config, "Configure file error"
resizeImageHeigh = int(resizeImageHeigh_config)

def show_image_with_path(image_path):
    # Read the image from file
    img = cv2.imread(image_path)
    # Check if the image was read successfully
    if img is not None:
        show_image_source(img, "Image")
    else:
        print(f"Error: Unable to load image from '{image_path}'")


def show_image_source(source, title = "Image"):
    cv2.imshow("Image", source)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close all OpenCV windows


def plot_images(real_image_paths, ai_image_paths):
    fig, axes = plt.subplots(2, len(real_image_paths), figsize=(12, 6))
    
    # Plot images for the first row
    for i, ax in enumerate(axes[0]):
        img_path = real_image_paths[i]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (int(resizeImageHeigh_config), int(resizeImageWidth_config)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        ax.imshow(img)
        # ax.axis('off')
        ax.set_title("Real")
    
    # Plot images for the second row
    for i, ax in enumerate(axes[1]):
        img_path = ai_image_paths[i]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (int(resizeImageHeigh_config), int(resizeImageWidth_config)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        ax.imshow(img)
        # ax.axis('off')
        ax.set_title("AI")
    
    plt.tight_layout()
    plt.show()


def plot_image_histogram(real_image, ai_image):
    
    real_image = cv2.resize(real_image, (resizeImageHeigh, resizeImageWidth))
    ai_image = cv2.resize(ai_image, (resizeImageHeigh, resizeImageWidth))
    
    hsv_ai_image = cv2.cvtColor(ai_image, cv2.COLOR_BGR2HSV)
    hsv_real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2HSV)
    
    hue_hist_ai = cv2.calcHist([hsv_ai_image[:,:,0]], [0], None, [180], [0, 180])
    hue_hist_real = cv2.calcHist([hsv_real_image[:,:,0]], [0], None, [180], [0, 180])
    
    saturation_hist_ai = cv2.calcHist([hsv_ai_image[:,:,1]], [0], None, [256], [0, 256])
    saturation_hist_real = cv2.calcHist([hsv_real_image[:,:,1]], [0], None, [256], [0, 256])
    
    brightness_hist_ai = cv2.calcHist([hsv_ai_image[:,:,2]], [0], None, [256], [0, 256])
    brightness_hist_real = cv2.calcHist([hsv_real_image[:,:,2]], [0], None, [256], [0, 256])
    
    average_hue_real = np.mean(hue_hist_real)
    print("Average hue for Real image:", average_hue_real)
    average_hue_ai = np.mean(hue_hist_ai)
    print("Average hue for AI image:", average_hue_ai)
    diff_image = cv2.absdiff(hsv_ai_image, hsv_real_image)
    diff_image_mean = np.mean(diff_image)
    threshold = diff_image_mean  # Adjust as needed
    _, thresholded_diff = cv2.threshold(diff_image[0], threshold, 255, cv2.THRESH_BINARY)
    num_differences = cv2.countNonZero(thresholded_diff)
    print("Number of differing pixels in hue:", num_differences)
    print("Number of differing pixels / total in hue:", num_differences / (hsv_ai_image.shape[0]* hsv_ai_image.shape[0]) )
    
    
    average_saturation_real = np.mean(saturation_hist_real)
    print("Average Saturation for Real image:", average_saturation_real)
    average_saturation_ai = np.mean(saturation_hist_ai)
    print("Average Saturation for AI image:", average_saturation_ai)
    diff_image = cv2.absdiff(hsv_ai_image, hsv_real_image)
    diff_image_mean = np.mean(diff_image)
    threshold = diff_image_mean  # Adjust as needed
    _, thresholded_diff = cv2.threshold(diff_image[0], threshold, 255, cv2.THRESH_BINARY)
    num_differences = cv2.countNonZero(thresholded_diff)
    print("Number of differing pixels in saturation:", num_differences)
    print("Number of differing pixels / total in saturation:", num_differences / (hsv_ai_image.shape[0]* hsv_ai_image.shape[0]) )
    
    average_brightness_real = np.mean(brightness_hist_real)
    print("Average brightness for Real image:", average_brightness_real)
    average_brightness_ai = np.mean(brightness_hist_ai)
    print("Average brightness for AI image:", average_brightness_ai)
    diff_image = cv2.absdiff(hsv_ai_image, hsv_real_image)
    diff_image_mean = np.mean(diff_image)
    threshold = diff_image_mean  # Adjust as needed
    _, thresholded_diff = cv2.threshold(diff_image[0], threshold, 255, cv2.THRESH_BINARY)
    num_differences = cv2.countNonZero(thresholded_diff)
    print("Number of differing pixels in brightness:", num_differences)
    print("Number of differing pixels / total in brightness:", num_differences / (hsv_ai_image.shape[0]* hsv_ai_image.shape[0]) )
    
    
    
    plt.figure(figsize=(20,5))
    
    colors = ['blue', 'green', 'red']
    channels = ['Hue', 'Saturation',  'Brightness']
    plt.subplot(1, 3, 1)
    plt.plot(hue_hist_ai, color=colors[0], label="AI")
    plt.plot(hue_hist_real, color=colors[2], label="Real")
    plt.xlabel(channels[0])
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(saturation_hist_ai, color=colors[0], label="AI")
    plt.plot(saturation_hist_real, color=colors[2], label="Real")
    plt.xlabel(channels[1])
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.plot(brightness_hist_ai, color=colors[0], label="AI")
    plt.plot(brightness_hist_real, color=colors[2], label="Real")
    plt.xlabel(channels[2])
    plt.ylabel('Frequency')
    
    plt.show()

            
def _test():
    rawDataPath_config = gl.App.get_config(section='IMAGE', option='rawDataPath', fallback=None)
    assert rawDataPath_config, "Configure file error"
    raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , str(rawDataPath_config)))
    imagePath = raw_data_path + r"\imageFx_Imagen2\ai_apple_2024-03-06\a real apple with a background of nothing\image_fx_a_real_apple_with_a_background_of_nothing_nat.jpg"
    show_image_with_path(imagePath)


def main():
    rawDataPath_config = gl.App.get_config(section='IMAGE', option='rawDataPath', fallback=None)
    assert rawDataPath_config, "Configure file error"
    raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , str(rawDataPath_config)))
    imagePath = raw_data_path + r"\imageFx_Imagen2\ai_apple_2024-03-06\a real apple with a background of nothing\image_fx_a_real_apple_with_a_background_of_nothing_nat.jpg"
    show_image_with_path(imagePath)            
        
        
if __name__ == '__main__':
    _test()