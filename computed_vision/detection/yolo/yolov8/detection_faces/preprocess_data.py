import os
import numpy as np
import pandas as pd
import shutil
import cv2
import random
import matplotlib.pyplot as plt
import copy
import wandb

bs=' ' # blank-space
class_id=0 # id for face
newline='\n' # new line character
extension='.txt' # extension for text file

# Creating paths for separate images and labels
curr_path=os.getcwd()
imgtrainpath = os.path.join(curr_path,'images','train')
imgvalpath=os.path.join(curr_path,'images','validation')
imgtestpath=os.path.join(curr_path,'images','test')

labeltrainpath=os.path.join(curr_path,'labels','train')
labelvalpath=os.path.join(curr_path,'labels','validation')
labeltestpath=os.path.join(curr_path,'labels','test')

data_path='./archive'

labels_path = os.path.join(curr_path, 'face_labels')

os.makedirs(labels_path, exist_ok=True)

list_archives = os.listdir(data_path)

print(list_archives)
# Defining input images and raw annotations path
img_path=os.path.join(data_path, 'images')
raw_annotations_path=os.path.join(data_path, 'faces.csv')


face_list=os.listdir(img_path)

print(face_list[:5])

data_len=len(face_list)
print(data_len)

train_split=0.8
val_split=0.1
test_split=0.1

imgtrain_list=face_list[:int(data_len*train_split)]
imgval_list=face_list[int(data_len*train_split):int(data_len*(train_split+val_split))]
imgtest_list=face_list[int(data_len*(train_split+val_split)):]

print('train: ', len(imgtrain_list))
print('valid: ', len(imgval_list))
print('test: ', len(imgtest_list))

# function to extract basename from a file and add a different extension to it. 
def change_extension(file):
    basename=os.path.splitext(file)[0]
    filename=basename+extension
    return filename


labeltrain_list = list(map(change_extension, imgtrain_list)) 
labelval_list = list(map(change_extension, imgval_list)) 
labeltest_list = list(map(change_extension, imgtest_list)) 

raw_annotations=pd.read_csv(raw_annotations_path)
# print(raw_annotations)


# Cria novas colunas
raw_annotations['x_centre']=0.5*(raw_annotations['x0']+raw_annotations['x1'])
raw_annotations['y_centre']=0.5*(raw_annotations['y0']+raw_annotations['y1'])
raw_annotations['bb_width']=raw_annotations['x1']-raw_annotations['x0']
raw_annotations['bb_height']=raw_annotations['y1']-raw_annotations['y0']

raw_annotations['xcentre_scaled']=raw_annotations['x_centre']/raw_annotations['width']
raw_annotations['ycentre_scaled']=raw_annotations['y_centre']/raw_annotations['height']
raw_annotations['width_scaled']=raw_annotations['bb_width']/raw_annotations['width']
raw_annotations['height_scaled']=raw_annotations['bb_height']/raw_annotations['height']

print(len(raw_annotations['image_name'].unique()))

# Getting all unique images
imgs=raw_annotations.groupby('image_name') 

for image in imgs:
    img_df=imgs.get_group(image[0])
    basename=os.path.splitext(image[0])[0]
    txt_file=basename+extension
    filepath=os.path.join(labels_path, txt_file)
    lines=[]
    i=1
    for index,row in img_df.iterrows():
        if i!=len(img_df):
            line=str(class_id)+bs+str(row['xcentre_scaled'])+bs+str(row['ycentre_scaled'])+bs+str(row['width_scaled'])+bs+str(row['height_scaled'])+newline
            lines.append(line)
        else:
            line=str(class_id)+bs+str(row['xcentre_scaled'])+bs+str(row['ycentre_scaled'])+bs+str(row['width_scaled'])+bs+ str(row['height_scaled'])
            lines.append(line)
        i=i+1
    with open(filepath, 'w') as file:
        file.writelines(lines)


random_file=os.path.join(labels_path, os.listdir(labels_path)[4])
with open (random_file, 'r') as f:
    content=f.read()

# print(content)

def_size=640 # Image size for YOLOv8

# print(len(os.listdir(labels_path))) # Verifying all labels are created)

# function to move files from source to detination
def move_files(data_list, source_path, destination_path):
    i=0
    for file in data_list:
        filepath=os.path.join(source_path, file)
        dest_path=os.path.join(data_path, destination_path)

        if not os.path.isdir(dest_path):
            os.makedirs(dest_path, exist_ok=True)
        if not os.path.isfile(os.path.join(dest_path, file)):
            shutil.move(filepath, dest_path, exist_ok=True)
            i=i+1
    print("Number of files transferred:", i)


# function to resize the images and copy the resized image to destination
def move_images(data_list, source_path, destination_path):
    i=0
    for file in data_list:
        filepath=os.path.join(source_path, file)
        dest_path=destination_path
        
        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)
        finalimage_path=os.path.join(dest_path, file)

        img_resized=cv2.resize(cv2.imread(filepath), (def_size, def_size))
        cv2.imwrite(finalimage_path, img_resized)
        i=i+1
    print("Number of files transferred:", i)
  

move_images(imgtrain_list, img_path, imgtrainpath)

move_images(imgval_list, img_path, imgvalpath)

move_images(imgtest_list, img_path, imgtestpath)

move_files(labeltrain_list, labels_path, labeltrainpath)

move_files(labelval_list, labels_path, labelvalpath)

move_files(labeltest_list, labels_path, labeltestpath)

print(len(os.listdir(labels_path)))

ln_1='# Train/val/test sets'+newline
ln_2='train: ' +"'"+imgtrainpath+"'"+newline
ln_3='val: ' +"'" + imgvalpath+"'"+newline
ln_4='test: ' +"'" + imgtestpath+"'"+newline
ln_5=newline
ln_6='# Classes'+newline
ln_7='names:'+newline
ln_8='  0: face'
config_lines=[ln_1, ln_2, ln_3, ln_4, ln_5, ln_6, ln_7, ln_8]


# Creating path for config file
config_path=os.path.join(curr_path, 'config.yaml')


# Writing config file
with open(config_path, 'w') as f:
  f.writelines(config_lines)


# function to obtain bounding box  coordinates from text label files
def get_bbox_from_label(text_file_path):
  bbox_list=[]
  with open(text_file_path, "r") as file:
      for line in file:
          _,x_centre,y_centre,width,height=line.strip().split(" ")
          x1=(float(x_centre)+(float(width)/2))*def_size
          x0=(float(x_centre)-(float(width)/2))*def_size
          y1=(float(y_centre)+(float(height)/2))*def_size
          y0=(float(y_centre)-(float(height)/2))*def_size
          
          vertices=np.array([[int(x0), int(y0)], [int(x1), int(y0)], 
                              [int(x1),int(y1)], [int(x0),int(y1)]])
#             vertices=vertices.reshape((-1,1,2))
          bbox_list.append(vertices)      
          
  return tuple(bbox_list)

# defining red color in RGB to draw bounding box
red=(255,0,0) 

plt.figure(figsize=(30,30))
for i in range(1,8,2):
  k=random.randint(0, len(imgtrain_list)-1)
  img_path=os.path.join(imgtrainpath, imgtrain_list[k])
  label_path=os.path.join(labeltrainpath, labeltrain_list[k])
  bbox=get_bbox_from_label(label_path)
  image=cv2.imread(img_path)
  image_copy=copy.deepcopy(image)
  ax=plt.subplot(4, 2, i)
  plt.imshow(image) # displaying image
  plt.xticks([])
  plt.yticks([])
  cv2.drawContours(image_copy, bbox, -1, red, 2) # drawing bounding box on copy of image
  ax=plt.subplot(4, 2, i+1)
  plt.imshow(image_copy) # displaying image with bounding box
  plt.xticks([])
  plt.yticks([])

plt.show()