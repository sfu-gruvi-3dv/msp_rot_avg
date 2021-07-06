import os
import lmdb
import shutil
import numpy as np
import sys
import cv2
import pickle

from core_io.lmdb_writer import LMDBWriter

class ImageLMDBWriter(LMDBWriter):
    
    __meta_data__ = None
    __meta_path__ = None
    
    def __init__(self, lmdb_path, meta_path, earse_exist=False, auto_start=True):
        super(ImageLMDBWriter, self).__init__(lmdb_path, earse_exist, auto_start)
        
        self.__meta_path__ = meta_path
        
        if earse_exist:
            self.__meta_data__ = dict()
        else:
            self.__meta_data__ = pickle.load(open(meta_path, "rb"))
            
    def __del__(self):
        super(ImageLMDBWriter, self).__del__()
        pickle.dump(self.__meta_data__, open(self.__meta_path__, "wb"))
        
    
    def insert_new_string(self, key, s):
        self.write_str(key,s)
        
    def insert_new_image_with_downsample(self, img_path, key, downsample_scale=0.5):
        img = cv2.imread(img_path)
        oriH, oriW = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (int(img.shape[1]*downsample_scale),
                               int(img.shape[0]*downsample_scale)))
        newH, newW = img.shape[:2]
        self.write_array(key, img.astype(np.uint8))
        self.__meta_data__[key] = {'OriDim': (oriH, oriW), 'dim': (newH, newW)}
        
    
        
    