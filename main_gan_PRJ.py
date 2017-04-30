
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
from PIL import Image

import scipy
from scipy import misc

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
        
    epochs = 9 #int(model["epochs"])
    batch_size = 128  #int(model["batch_size"])
    chunk_size = batch_size
    sim_dict_lgth = 5

    slice_data = 80384

    path_to_matrix_sim_1hot= str(model["matrix_sim_1hot"])
    path_To_Params = str(model["path_To_Params"])
    
    if path_To_Params == "None":
        path_To_Params = None

    global train_paths
    global dev_paths

    train_paths = model["data"]["train"]
    valid_paths = model["data"]["valid"]
    
    train_data = np.float32(np.load(train_paths["images"]))
    
    train_data_captions = np.float32(np.load(train_paths["captions"]))
    
    
    val_data = np.float32(np.load(valid_paths["images"]))   
    
    val_data_captions = np.float32(np.load(valid_paths["captions"]))
    
    mu, sigma = 0, 1./36000 # mean and standard deviation

    matrix_sim_1hot = np.random.normal(mu, sigma, (1,sim_dict_lgth))
    
    '''
    il faut la load la matrice matrix_sim_1hot
    '''
    #matrix_sim_1hot = np.load(path_to_matrix_sim_1hot)


    val_data_captions=val_data_captions.reshape(val_data_captions.shape[0],5,62,1)

    val_data_captions=np.float32(np.dot(val_data_captions,matrix_sim_1hot))

    train_data_captions=train_data_captions.reshape(train_data_captions.shape[0],5,62,1)

    train_data_captions=np.float32(np.dot(train_data_captions,matrix_sim_1hot))
    
    ''' Saving the similator of one hot transformor '''
    np.save(path_to_matrix_sim_1hot,matrix_sim_1hot)    

        
    train_data=np.float32(train_data[:slice_data])
    val_data=np.float32(val_data[:(slice_data//2)])
    
    train_data_captions=train_data_captions[:slice_data]
    val_data_captions=val_data_captions[:(slice_data//2)]
    
    
    num_chunk_train = train_data.shape[0]//chunk_size
    num_chunk_val = val_data.shape[0]//chunk_size
    
    epsilon_dis = 0.000001
    epsilon_gen = 0.000001
    cost_vl_best = float('Inf')
    
    numpy_rng = np.random
    MOD_OPTM = MODEL_OPTIMIZER(batch_size, epsilon_dis, epsilon_gen,num_filters_g=[64,128,256,512],\
                num_filters_d=[32,64,64,128,256,256,256,512,1], \
                               sim_dict_lgth = sim_dict_lgth, X_words_lgth=62,Z_words_lgt=10,Z_ctxt_lgt=90)#35631
    
            
    #num_filters_g=[8,16,32,64]
    #num_filters_d=[4,8,16,32,64,64,64,64,1]
    #num_filters_g=[16,32,64,128]
    #num_filters_d=[8,16,32,64,128,128,128,128,1]
    #num_filters_g=[64,128,256,512]
    #num_filters_d=[32,64,128,256,512,512,512,512,1]
    print('train images shape: ',train_data.shape)
    print('val images shape: ',val_data.shape)
    print('train captions shape: ',train_data_captions.shape)
    print('val captions shape: ',val_data_captions.shape)
    
    print('chunk_size: ',chunk_size)
    print('num_chunk_train: ',num_chunk_train)
    print('num_chunk_val: ',num_chunk_val)
    
    train_chunk = theano.shared(np.asarray(train_data[0:chunk_size],dtype='float32'), borrow=True)
    val_chunk = theano.shared(np.asarray(val_data[0:chunk_size],dtype='float32'), borrow=True)
    
    
    
    train_caption_chunk = theano.shared(np.asarray(train_data_captions[0:chunk_size],dtype='float32'), borrow=True)
    val_caption_chunk = theano.shared(np.asarray(val_data_captions[0:chunk_size],dtype='float32'), borrow=True)


    #discriminator_update, generator_update, get_valid_cost = MOD_OPTM.optimize_gan(train_set, val_set, lam1=0.00001)
    discriminator_update, generator_update,  get_valid_cost  = MOD_OPTM.optimize_gan(train_chunk, val_chunk,\
                                                                  train_caption_chunk,val_caption_chunk, lam1=0.00001)

    #params=np.load(path_To_Params)
        
    #print('..setting weights...')        
        
    #lasagne.layers.set_all_param_values(MOD_OPTM.layer_gen_save, params)
    eps_gen = 0.0000001
    eps_dis = 0.0000001
    #1e-15 #get_epsilon(epsilon_dis, 25, epoch)#get_epsilon_inc(epsilon_dis, 25, epoch) 
    

    batch_in_chunk_train = chunk_size//batch_size
    batch_in_chunk_val = chunk_size//batch_size
    
    print ('...Start Training')
  
    for epoch in range(epochs+1):

        costs=[[],[], []]
        exec_start = timeit.default_timer()
               
        for chunk_train_i in range(num_chunk_train):
            train_chunk.set_value(train_data[chunk_train_i*chunk_size:(chunk_train_i+1)*chunk_size], borrow=True)
            train_caption_chunk.set_value(train_data_captions[chunk_train_i*chunk_size:(chunk_train_i+1)*chunk_size], borrow=True)
            
            for batch_i in range(batch_in_chunk_train):
                
                for caption_i in range(5):
                    cost_disc_i = discriminator_update(batch_i,caption_i, lr=eps_dis)                
                    cost_gen_i = generator_update(batch_i,caption_i, lr=eps_gen)
                    #while np.isnan(cost_disc_i):
                    #    eps_dis=eps_dis/10.
                    #    cost_disc_i = discriminator_update(batch_i,caption_i, lr=eps_dis)
                    costs[0].append(cost_disc_i)   
                    costs[1].append(cost_gen_i) 
                
            if (chunk_train_i%128==0): print ('chunk number process: ',chunk_train_i)

        exec_finish = timeit.default_timer() 
        if epoch==0: print ('Exec Time %f ' % ( exec_finish - exec_start))

        
        if epoch >=0:

            costs_vl = [[],[],[]]
            for chunk_val_j in range(num_chunk_val):
                original_images=val_data[chunk_val_j*chunk_size:(chunk_val_j+1)*chunk_size]
                val_chunk.set_value(original_images, borrow=True)
                val_caption_chunk.set_value(val_data_captions[chunk_val_j*chunk_size:(chunk_val_j+1)*chunk_size], borrow=True)
                
                for batch_j in range(batch_in_chunk_val):
                    
                    for caption_j in range(5):
                        cost_dis_vl_j,cost_gen_vl_j , gen_images,Orig_images = get_valid_cost(batch_j,caption_j)
                        costs_vl[0].append(cost_dis_vl_j)
                        costs_vl[1].append(cost_gen_vl_j)
                    
            cost_dis_vl = np.mean(np.asarray(costs_vl[0]))
            cost_gen_vl = np.mean(np.asarray(costs_vl[1]))             

            cost_dis_tr = np.mean(np.asarray(costs[0]))
            cost_gen_tr = np.mean(np.asarray(costs[1]))

            cost_tr = cost_dis_tr + cost_gen_tr
            cost_vl = cost_dis_vl + cost_gen_vl
            
            if cost_vl < cost_vl_best:
                print('..saving weights best val...')                
                gen_images_best=gen_images
                Orig_images_best=Orig_images
                np.save(path_To_Params,lasagne.layers.get_all_param_values(MOD_OPTM.layer_gen_save))
                
                cost_vl_best = cost_vl                
                
                #np.save("/home/didier/AAA_IFT6266/Class_Project/AA_GAN_PRJ_CODE/WEIGHTS_PRJ/generation_weights.npy",\
                       # all_param_values_gen)#params_values_gen
            '''       
            elif np.isinf(cost_vl_best):
                eps_dis=eps_dis   
            elif np.isnan(cost_vl):
                eps_dis=eps_dis/10.
            else:    
                eps_dis=eps_dis/5.
            '''   
               
            #print ('Epoch %d, epsilon_gen %f5, epsilon_dis %f5, tr disc gen %g, %g | vl disc gen %g, %g '\
            #        % (epoch, eps_gen, eps_dis, cost_dis_tr, cost_gan_tr, cost_dis_vl, cost_gan_vl))
            print ('Epoch %d , epsilon_dis %f5 , epsilon_gen %f5 , tr sum %g |  vl sum %g'\
                    % (epoch, eps_dis,eps_gen,cost_tr, cost_vl))
    
    for i in range(15) :
        #pylab.subplot(1, 15, i+1); pylab.axis('off'); pylab.imshow(gen_images_best[i])
        #im = Image.fromarray(gen_images_best[i],'RGB')
        #im.save("/home/didier/AAA_IFT6266/Class_Project/AA_GAN_PRJ_CODE/MODELS_PRJ/images"+str(i)+".png") 
        misc.imsave("/home2/ift6ed45/AA_DCGAN_3_PRJ_CODE/MODELS_PRJ/images"+str(i)+".png",gen_images_best[i])
        misc.imsave("/home2/ift6ed45/AA_DCGAN_3_PRJ_CODE/MODELS_PRJ/images_orig"+str(i)+".png",Orig_images_best[i])

    '''    
    for i in range(5) :
        pylab.subplot(2, 5, i+1); pylab.axis('off'); pylab.imshow(gen_images_best[i+5])
        
    for i in range(5) :
        pylab.subplot(3, 5, i+1); pylab.axis('off'); pylab.imshow(gen_images_best[i+10])
    '''
    #pylab.savefig('/home/didier/AAA_IFT6266/Class_Project/AA_GAN_PRJ_CODE/MODELS_PRJ/images.png')
    #pylab.show()
    
    
    
