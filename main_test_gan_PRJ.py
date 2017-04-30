import theano
import theano.tensor as T
import numpy as np
theano.config.floatX = 'float32'


if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams 


from collections import OrderedDict
import time, timeit
import datetime
import sys
import json
import math
import pickle as pickle
import pylab
from model_optimizer_PRJ import *
from utils_PRJ import * 

debug = sys.gettrace() is not None
if debug:
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'
    theano.config.compute_test_value = 'warn'

theano.config.scan.allow_gc = True
theano.config.allow_gc = True
 

assert(theano.config.scan.allow_gc == True), "set scan.allow_gc to True ; otherwise you will run out of gpu memory"
assert(theano.config.allow_gc == True), "set allow_gc to True ; otherwise you will run out of gpu memory"

sys.stdout.flush()

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30)) #np.random.randint(1 << 30)

if __name__ == '__main__':

    model_name = sys.argv[1]
    with open(model_name) as model_file:
        model = json.load(model_file)

        
    batch_size = 128  #int(model["batch_size"])
    
    path_to_matrix_sim_1hot= str(model["matrix_sim_1hot"])
    path_To_Params = str(model["path_To_Params"])
    
    if path_To_Params == "None":
        print()
        print("  Provide path to parameters and test again.")
        print()
        
    else:
        test_paths = model["data"]["test"]
    
        test_data = np.float32(np.load(test_paths["images"]))
    
        test_data_captions = np.float32(np.load(test_paths["captions"]))
        '''
        il faut la load la matrice matrix_sim_1hot
        '''
        matrix_sim_1hot = np.load(path_to_matrix_sim_1hot)
        
        test_data=test_data[:128]#[32770:32898]
    
        test_data_captions=test_data_captions[:128]#[32770:32898]

        #matrix_sim_1hot = np.float32(np.load(path_to_matrix_sim_1hot))
        
        test_data_captions_resh=test_data_captions.reshape((test_data_captions.shape[0],1,62,1))

        test_data_captions_resh=np.dot(test_data_captions_resh,matrix_sim_1hot)

    
        num_batch_test = test_data.shape[0]//batch_size
        
        sim_dict_lgth = 5

        MOD_OPTM = MODEL_OPTIMIZER(batch_size,num_filters_g=[8,16,32,64],\
                num_filters_d=[8,16,32,64,128,128,128,128,1], sim_dict_lgth = sim_dict_lgth, X_words_lgth=62)#35632
    
        print('test data shape: ',test_data.shape)
        print('test captions shape: ',test_data_captions_resh.shape)
        print('num_batch_test: ',num_batch_test)
        
        test_set = theano.shared(np.asarray(test_data,dtype='float32'))
        
        test_caption_set = theano.shared(np.asarray(test_data_captions_resh,dtype='float32'))
        
        get_test_images  = MOD_OPTM.test_gan(test_set,test_caption_set)
    
        print ('...Start Testing')

        images=[]#np.zeros((num_batch_test, batch_size, 64, 64, 3))
        
        exec_start = timeit.default_timer()
        
        params=np.load(path_To_Params)
        
        print('..setting weights...')        
        
        lasagne.layers.set_all_param_values(MOD_OPTM.layer_gen_save, params)
        
                
        for batch_i in range(num_batch_test):
            
            image = get_test_images(batch_i)
            
            images.append(image)
             
            #images[batch_i] = image
                
                
        exec_finish = timeit.default_timer() 
        
        print ('Exec Time %f ' % ( exec_finish - exec_start))
                
        np.save("/home/didier/AAA_IFT6266/Class_Project/AA_GAN_PRJ_CODE/WEIGHTS_PRJ/gen_images.npy",\
                images)  
        print("image shape : ",image[0][0].shape)
        pylab.subplot(1, 1, 1); pylab.axis('off'); pylab.imshow(image[0][0])
        pylab.show()
        
        '''
        gen_images.shape 
        (num_batch_test, 1, batch_size, 64, 64, 3)
        gen_images_t=np.transpose(gen_images,(1,0,2,3,4,5))
        gen_images_t[0].shape
        (num_batch_test, batch_size, 64, 64, 3)
        pylab.subplot(1, 1, 1); pylab.axis('off'); pylab.imshow(gen_images_t[0][0][0])
        pylab.show()
        '''
        print ('THE END')
         
    
    
