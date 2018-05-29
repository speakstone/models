# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:05:39 2018

@author: lilei0129
"""

import cv2
import os 
from PIL import Image   
import numpy as np
  
def MIoU(path1,path2, change):  #传入存储的list
    MIoU_list = np.array([])
    for file in os.listdir(path1):
            name =  os.path.join( file) 
            file_name = list(name)
            name_index = file_name.index('.')
            file_name = file_name[:name_index]
            file_name = ''.join(file_name)
            test_img = file_name+"_seg_color.jpg"
            file_img_name = os.path.join( path1,name)            
            test_img_name = os.path.join( path2,name)
            file_img = cv2.imread(file_img_name,0)
            test_img = cv2.imread(test_img_name,0)
            file_img_list = np.array(file_img).astype('bool').astype('int')
            test_img_list = np.array(test_img).astype('bool').astype('int')           
            Convergence = file_img_list*test_img_list
            Intersection = np.array(file_img_list+test_img_list).astype('bool').astype('int')
            MIoU_Sum = (np.sum(Convergence))/(np.sum(Intersection))
            MIoU_list = np.concatenate((MIoU_list,[MIoU_Sum])) # 先将p_变成list形式进行拼接，注意输入为一个tuple
            MIoU_list = np.append(MIoU_list,MIoU_Sum) #直接向p_arr里添加p_
    return MIoU_list        
    

            
#            seg_name = file_name+'.png'
#            file_name = file_name+'.jpg'
#            file_name = os.path.join( change,file_name)
#            seg_name = os.path.join( path2,seg_name)
#            print file_name
#            file_path = os.path.join( path1,file)  
#            im = cv2.imread(file_path)
#            cv2.imwrite(file_name,im)
            
            
if __name__ == '__main__':
#  change_name(path1,path2,path3)
  change_size(path3,path2,data1)
