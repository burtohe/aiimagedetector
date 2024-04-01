import os
import cv2
import imagemanagement as imagemanager


def ela(image, dst_folder, quality = 95, scale = 15):
    # write img1 at 95% jpg compression
    temp_file = "image_compressed.jpg"
    temp_ela_file = "image_ela.png"
    des_folder_temp = os.path.join(dst_folder, temp_file)
    des_folder_new = os.path.join(dst_folder, temp_ela_file)
    
    cv2.imwrite(des_folder_temp, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    # read compressed image
    compressed_image = cv2.imread(des_folder_temp)
    # get absolute difference between img1 and img2 and multiply by scale
    diff1 = scale * cv2.absdiff(image, compressed_image)
    os.remove(des_folder_temp)
    # cv2.imwrite(des_folder_new, diff1) 
    # os.remove(des_folder_new)
    return diff1

    
    

def _test():
    pretraining_data_path = imagemanager.get_data_path('IMAGE', 'preTrainingDataPath', None, '..')
    assert pretraining_data_path, "Configure file error"
    image = cv2.imread("reconstructed_images.png")
    ela_result = ela(image, pretraining_data_path, quality = 95, scale = 15)
    # temp_ela_file = "image_ela.png"
    # des_folder_new = os.path.join(pretraining_data_path, temp_ela_file)
    # os.remove(des_folder_new)
    cv2.imshow("ela95", ela_result)
    cv2.waitKey(0)

if __name__ == '__main__':
    _test()


# # read image
# img1 = cv2.imread("lenna.png")

# # set compression and scale
# jpg_quality1 = 95
# jpg_quality2 = 90
# scale = 15

# # write img1 at 95% jpg compression
# cv2.imwrite("lenna_c95.jpg", img1, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality1])

# # read compressed image
# img2 = cv2.imread("lenna_c95.jpg")

# # get absolute difference between img1 and img2 and multiply by scale
# diff1 = scale * cv2.absdiff(img1, img2)

# # write img2 at 90% jpg compression
# cv2.imwrite("lenna_c90.jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality2])

# # read compressed image
# img3 = cv2.imread("lenna_c90.jpg")

# # get absolute difference between img1 and img2 and multiply by scale
# diff2 = scale * cv2.absdiff(img2, img3)

# # write result to disk
# cv2.imwrite("lenna_ela_95.jpg", diff1)
# cv2.imwrite("lenna_ela_90.jpg", diff2)

# # display it
# cv2.imshow("ela95", diff1)
# cv2.imshow("ela90", diff2)
# cv2.waitKey(0)