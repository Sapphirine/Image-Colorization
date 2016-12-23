# Image-Colorization

Image Colorization contains the following files 

vgg16_mod.py - The model for the CNN that is trained for image colorization
bigdata_final_01.m - Takes all the jpg files in the mirflickr folder and does image processing.
image_color.py - Takes the processed matlab arrays from bigdata_final_01.m , creates training and testing sets, and trains the NN. 
colorize.py - Does prediction. Feeds the intensity channel of the image, and returns two color channels. 
bigdata_final_01.m - Reconstructs the image after by taking the output of colorize.py
