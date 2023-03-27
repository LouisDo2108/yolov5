import os
import cv2

# specify the path to the image folder
folder_path = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/val/images"
output_paths = [
    "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/blur_10",
    "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/blur_20",
    "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/blur_30",
    "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/blur_40"
]

# define a list of Gaussian blur levels to apply
blur_levels = [(3, 3), (7, 7), (11, 11), (15, 15)]

# loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        # read in the image
        image_path = os.path.join(folder_path, file_name)
        img = cv2.imread(image_path)

        # apply each level of Gaussian blur and save the resulting images
        for ix, ksize in enumerate(blur_levels):
            img_blur = cv2.GaussianBlur(img, ksize, sigmaX=0)
            output_path = os.path.join(output_paths[ix], f"{file_name[:-4]}.jpg")
            cv2.imwrite(output_path, img_blur)