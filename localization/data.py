# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

class CUBDataset( object ):

    def __init__( self ):

        self._img_num   = 11788
        self._out_dim   = 4
        self._row       = 224
        self._column    = 224
        self._dataset_path  = '/data/zy/course/cv/localization/'
        
        with open( self._dataset_path +'data/images.txt') as f:
            id_to_path = dict()
            for l in f.read().splitlines():
                im_id, im_path = l.split( ' ', 1 )
                id_to_path[ int(im_id) ] = im_path
        
        with open( self._dataset_path + 'data/bounding_boxes.txt') as f:
            id_to_box = dict()
            for line in f.read().splitlines():
                im_id, *box = line.split(' ')
                id_to_box[ int(im_id) ] = list(map(float, box))

        # import pdb; pdb.set_trace()
        self._imgs = [(os.path.join( self._dataset_path + 'data/images', id_to_path[i]), id_to_box[i]) for i in range( 1, self._img_num + 1 )]

            
    def get_item(self, index):
        
        # import pdb; pdb.set_trace()
        path, box   = self._imgs[index]
        im          = Image.open( path )

        im_size = np.array(im.size, dtype='float32')
        box = np.array(box, dtype='float32')
        
        im  = im.resize( (224,224) ).convert( 'RGB' )
        im  = np.asarray( im )

        box     = np.divide( 224 * box, [ im_size[0], im_size[1], im_size[0], im_size[1] ] )

        # normalization
        mean    = np.array([0.485, 0.456, 0.406])
        std     = np.array([0.229, 0.224, 0.225])
        im = ( im / 255 - mean ) / std
        
        return im, box


if __name__ == '__main__':

    # for test
    dataset = CUBDataset( )
    dataset.get_item( 3 )