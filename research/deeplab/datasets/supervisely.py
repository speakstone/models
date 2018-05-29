# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:02:22 2018

@author: lilei0129
""" 

import cv2
import os  
import numpy as np
  
def change_content(path1,path2, change):  #传入存储的list
	for file in os.listdir(path1):
            name =  os.path.join( file) 
            file_name = list(name)
            name_index = file_name.index('.')
            file_name = file_name[:name_index]
            file_name = ''.join(file_name)
            seg_name = file_name+'.png'
            file_name = file_name+'.jpg'
            file_name = os.path.join( change,file_name)
            seg_name = os.path.join( path2,seg_name)
            print file_name
            file_path = os.path.join( path1,file)  
            im = cv2.imread(file_path)
            cv2.imwrite(file_name,im)


def change_name(path1): 
    for file in os.listdir(path1):
            name =  os.path.join( file) 
            name = list(name)
            name_index = name.index('.')
            name = name[:name_index]
            name = ''.join(name)
            old_name = os.path.join( path1,file)
            new_name = os.path.join( path1 , name+'.jpg')
            os.rename(old_name,new_name)


def change_size(path1,path2, list_name):
	for file in os.listdir(path1):
            name =  os.path.join( file) 
            name = list(name)
            name_index = name.index('.')
            name = name[:name_index]
            name = ''.join(name)
            seg_name = name+'.png'
            seg_name =  os.path.join( path2,seg_name)  
            file_name = name+'.jpg'
            file_name =  os.path.join( path1,file_name) 
            seg_im = cv2.imread(seg_name,0)
            file_im = cv2.imread(file_name,0)

            sp1 = seg_im.shape
            sp2 = file_im.shape
            if sp1 != sp2:    
                sp = [name,sp1,sp2]
                                
                print sp
                os.remove(seg_name)
                os.remove(file_name)
#                sp_change =  sp2[::-1]
#                res=cv2.resize(seg_im,sp_change)
#                cv2.imwrite(seg_name,res)
                
#                print sp
            else:
                continue
            
#            if os.path.isdir(file_name):
#                  os.listdir(file_name, list_name) 
#            else:
#                  list_name.append(file_name)
def change_label(path1,path2):
    for file in os.listdir(path1):
            name =  os.path.join( file) 
            seg_name =  os.path.join( path1,name)
            print_name = os.path.join( path2,name)
            seg_im = cv2.imread(seg_name,0)
            seg_im = np.array(seg_im).astype('bool').astype('int')
            cv2.imwrite(print_name,seg_im)

                		
path1 = "/opt/data1/lilei/Supervisely/people_segment/Images/train/"
path2 = "/opt/data1/lilei/Supervisely/people_segment/Segments/train/"
path3 = "/opt/data1/lilei/Supervisely/people_segment/Image_change/train/"
path4 = "/opt/data1/lilei/Supervisely/people_segment/Segments_change/train/"
data1 = []
data2 = []


if __name__ == '__main__':
#  change_name(path1,path2,path3)
#  change_size(path3,path2,data1)
    change_label(path2,path4)
#  change_name(path3)
#  listdir(path2,data2)
#  print data1
