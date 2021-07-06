import os
import lmdb
import shutil
import numpy as np
import sys
import cv2
import pickle

from core_io.lmdb_reader import LMDBModel

class ImageLMDBModel(LMDBModel):
    
    __meta_data__ = None
    __meta_path__ = None
    
    def __init__(self, lmdb_path, meta_path):
        super(ImageLMDBModel, self).__init__(lmdb_path)
        
        self.__meta_path__ = meta_path
        self.__meta_data__ = pickle.load(open(meta_path, "rb"))
        
    def __del__(self):
        super(ImageLMDBModel, self).__del__()
        
    def getImageAndDimByKey(self, key):
        print(key)
        img = self.read_ndarray_by_key(key, dtype=np.uint8)
        
        img_meta = self.__meta_data__[key]
        downsample_scale = img_meta['dim'][0] / img_meta['OriDim'][0]
        ori_img_dim = img_meta['OriDim']
        img_dim = img_meta['dim']

        img = img.reshape(int(img_dim[0]), int(img_dim[1]), 3)
        
        return img, ori_img_dim
        