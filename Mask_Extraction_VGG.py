import os
import cv2
import json
import numpy as np
import argparse


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('srcdir',  help='file directory', type=str)
  parser.add_argument('json', help='json file name', type=str)
  args = parser.parse_args()
  return args

def json_read(path):
  with open(path) as f:
    data = json.load(f)
  return data

def add_to_dict(data, itr, key, count):
    try:
        x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
    except:
        print("No BB. Skipping", key)
        return
    
    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])
    
    file_bbs[key] = all_points

    return file_bbs

def extract(data, source_folder, file_bbs, count, mask_width, mask_height):

  for itr in data:
    file_name_json = data[itr]["filename"]
    sub_count = 0               # Contains count of masks for a single ground truth image
    
    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            key = file_name_json[:-4] + "*" + str(sub_count+1)
            file_bbs = add_to_dict(data, itr, key, sub_count)
            sub_count += 1
    else:
        file_bbs = add_to_dict(data, itr, file_name_json[:-4], 0)

  print("\nDict size: ", len(file_bbs))

  for file_name in os.listdir(source_folder):
      print(file_name)
      to_save_folder = os.path.join(source_folder, file_name[:-4])
      image_folder = os.path.join(to_save_folder, "images")
      mask_folder = os.path.join(to_save_folder, "masks")
      curr_img = os.path.join(source_folder, file_name)
    
      # make folders and copy image to new location
      os.mkdir(to_save_folder)
      os.mkdir(image_folder)
      os.mkdir(mask_folder)
      os.rename(curr_img, os.path.join(image_folder, file_name))

  for itr in file_bbs:
    num_masks = itr.split("*")
    to_save_folder = os.path.join(source_folder, num_masks[0])
    mask_folder = os.path.join(to_save_folder, "masks")
    mask = np.zeros((mask_width, mask_height))
    try:
        arr = np.array(file_bbs[itr])
    except:
        print("Not found:", itr)
        continue
    count += 1
    cv2.fillPoly(mask, [arr], color=(255))
    
    if len(num_masks) > 1:
    	cv2.imwrite(os.path.join(mask_folder, itr.replace("*", "_") + ".jpg") , mask)    
    else:
        cv2.imwrite(os.path.join(mask_folder, itr + ".png") , mask)



if __name__ == '__main__':
  args = parse_args()
  
  # Initialization
  img_path = os.path.join(args.srcdir,"Images")
  json_path = os.path.join(args.srcdir,args.json)
  count = 0
  file_bbs = {}
  mask_width = 2976
  mask_height = 2976

  data = json_read(json_path)
  extract(data, img_path, file_bbs, count, mask_width, mask_height)
  