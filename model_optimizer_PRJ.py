import theano 
import numpy as np
import scipy as sp
import lasagne
from lasagne.regularization import regularize_network_params, l2 
from generator_network import*
from discriminator_network import *
theano.config.floatX = 'float32'

class MODEL_OPTIMIZER():

    def __init__(self,batch_sz, epsilon_dis=0.0000002,epsilon_gen=0.0000002,num_filters_g=[8,16,32,64],\
                num_filters_d=[4,8,16,32,64,64,64,64,1], \
                 sim_dict_lgth=10, X_words_lgth=62,Z_words_lgt=10,Z_ctxt_lgt=90):
        """Initializes discriminator and generator and optimise their loss functions"""
        
        self.epsilon_dis = epsilon_dis
        self.epsilon_gen = epsilon_gen
        self.sim_dict_lgth= sim_dict_lgth
        self.X_words_lgth=X_words_lgth
        self.batch_sz = batch_sz
        self.num_filters_g = num_filters_g
        
        self.dis_network = DISCRIMINATOR_NET(self.batch_sz , num_filters_d) 
        self.gen_network = GENERATOR_NET(self.batch_sz , num_filters_g, sim_dict_lgth=self.sim_dict_lgth,\
                                         X_words_lgth=self.X_words_lgth)
        
        mat_to_Z_words = np.float32(np.random.normal(0, 1, (sim_dict_lgth*X_words_lgth,Z_words_lgt)))
        self.mat_to_Z_words_shd = theano.shared(mat_to_Z_words, name='mat_to_Z_words_shd')
        
        mat_to_Z_ctxt = np.float32( np.random.normal(0, 1, (3*64*64,Z_ctxt_lgt)))
        self.mat_to_Z_ctxt_shd = theano.shared(mat_to_Z_ctxt, name='mat_to_Z_ctxt_shd')

        mat_to_first_FeatMap_flat = np.float32( np.random.normal(0, 1, (Z_words_lgt + Z_ctxt_lgt,num_filters_g[3]*2*2)))
        self.mat_to_first_FeatMap_flat_shd = theano.shared(mat_to_first_FeatMap_flat,\
                                                           name='mat_to_first_FeatMap_flat_shd')

        
    def cost_dis_fn(self, X_ORIGINAL,X_SENTENCE,lam1):
        """compute cost of the discriminator"""
        
        X_ORIG_CTXT = theano.tensor.set_subtensor(X_ORIGINAL[:,:,16:48,16:48], 0.0)
        X_ORIG_32   = X_ORIGINAL[:,:,16:48,16:48]
        
        X_SENTENCE_flat = X_SENTENCE.flatten(2) 
        X_ORIG_CTXT_flat = X_ORIG_CTXT.flatten(2)
        
        Z_words = theano.tensor.dot(X_SENTENCE_flat,self.mat_to_Z_words_shd)
        Z_ctxt = theano.tensor.dot(X_ORIG_CTXT_flat,self.mat_to_Z_ctxt_shd) 
        
        Z = theano.tensor.concatenate([Z_words,Z_ctxt],axis=1)
        
        first_FeatMap = theano.tensor.dot(Z,self.mat_to_first_FeatMap_flat_shd)
        first_FeatMap = first_FeatMap.reshape((self.batch_sz, self.num_filters_g[3],2,2))
        
        
        '''Discriminate of the original image '''
        target1  = T.alloc(.9, self.batch_sz)
        l_in_d = lasagne.layers.InputLayer ( shape = (self.batch_sz , 3, 32,32), input_var = X_ORIG_32 )
        
        l_dis_flat = lasagne.layers.FlattenLayer(self.dis_network.propagate(l_in_d))
        
        #print("l_dis_flat shape", lasagne.layers.get_output_shape(l_dis_flat))
        
        p_y__x1  = lasagne.layers.get_output(l_dis_flat)
        
        p_y__x1  = p_y__x1.flatten()
        
        '''Discriminate of the generated image '''
        
        layer_gen_out        = self.gen_network.generate_image(first_FeatMap)#(X_ORIGINAL,X_SENTENCE)
        self.layer_gen_save  = layer_gen_out
        
          
        gen_images          = lasagne.layers.get_output(self.layer_gen_save)
        
        l_in_d_0            = lasagne.layers.InputLayer ( shape = (self.batch_sz , 3, 32,32), input_var = gen_images )
        
        layer_dis_out_0     = self.dis_network.propagate(l_in_d_0)
        
        '''les parametres du discriminator seuls'''
        params_dis          = lasagne.layers.get_all_params(layer_dis_out_0,trainable=True)
        
        target0             = T.alloc(0., self.batch_sz) 
        
        p_y__x0             = lasagne.layers.get_output(layer_dis_out_0)
        
        #print("layer_dis_out_0 shape", lasagne.layers.get_output_shape(layer_dis_out_0))
        
        p_y__x0             = p_y__x0.flatten() 
        
        lam_w_decay= lam1*regularize_network_params(layer=layer_dis_out_0, penalty=l2)
        
        cost = (T.mean(T.nnet.binary_crossentropy(p_y__x1, target1)) \
                       + T.mean(T.nnet.binary_crossentropy(p_y__x0, target0)))+lam_w_decay
                        
        return  cost, params_dis #, X_ctxt_gen.dimshuffle(0,2,3,1)




    def cost_gen_fn(self,X_ORIGINAL ,X_SENTENCE,lam1):
        """compute cost of the discriminator"""
        
        X_ORIG_CTXT = theano.tensor.set_subtensor(X_ORIGINAL[:,:,16:48,16:48], 0.0)
        X_ORIG_32   = X_ORIGINAL[:,:,16:48,16:48]
        
        X_SENTENCE_flat = X_SENTENCE.flatten(2) 
        X_ORIG_CTXT_flat = X_ORIG_CTXT.flatten(2)
        
        Z_words = theano.tensor.dot(X_SENTENCE_flat,self.mat_to_Z_words_shd)
        Z_ctxt = theano.tensor.dot(X_ORIG_CTXT_flat,self.mat_to_Z_ctxt_shd) 
        
        Z = theano.tensor.concatenate([Z_words,Z_ctxt],axis=1)
        
        first_FeatMap = theano.tensor.dot(Z,self.mat_to_first_FeatMap_flat_shd)
        first_FeatMap = first_FeatMap.reshape((self.batch_sz, self.num_filters_g[3],2,2))

        
        '''Discriminate of the generated image '''
        
        layer_gen_out        = self.gen_network.generate_image(first_FeatMap)#(X_ORIGINAL,X_SENTENCE)        
        params_gen           = lasagne.layers.get_all_params(layer_gen_out,trainable=True)
              
        target1             = T.alloc(.9, self.batch_sz)   
        gen_images          = lasagne.layers.get_output(layer_gen_out)
        l_in_d_0            = lasagne.layers.InputLayer ( shape = (self.batch_sz , 3, 32,32), input_var = gen_images )
        p_y__x1             = lasagne.layers.get_output(self.dis_network.propagate(l_in_d_0))
        p_y__x1             = p_y__x1.flatten() 
        
        lam_w_decay= lam1*regularize_network_params(layer=layer_gen_out, penalty=l2)
        
        X_ctxt_gen = theano.tensor.set_subtensor(X_ORIGINAL[:,:,16:48,16:48], gen_images) #(X_ORIG_CTXT[:,:,16:48,16:48], gen_images)
                        
        return (T.mean(T.nnet.binary_crossentropy(p_y__x1, target1)) )+lam_w_decay,\
                       params_gen, X_ctxt_gen.dimshuffle(0,2,3,1), X_ORIGINAL.dimshuffle(0,2,3,1)
        
  
        

    def optimize_gan(self, train_chunk, valid_chunk, train_caption_chunk, val_caption_chunk,\
                     lam1=0.000001):
    
 
        lr = theano.tensor.fscalar('lr');
        #drpout = theano.tensor.bscalar('drpout');
        X_batch = theano.tensor.tensor4(name='X_batch',dtype='float32');
        #X_batch_32 = theano.tensor.tensor4(name='X_batch_32',dtype='float32');
        #X_batch_ctxt = theano.tensor.tensor4(name='X_batch_ctxt',dtype='float32');
        X_words = theano.tensor.tensor4(name='X_words',dtype='float32');
        i = theano.tensor.scalar('i',dtype='int32');
        k = theano.tensor.scalar('k',dtype='int32');
        
        
        """compute cost of the discriminator"""
        
        cost_disc , params_dis = self.cost_dis_fn(X_batch,X_words,lam1) 
        
        cost_gen , params_gen, gen_images_gen, Orig_images = self.cost_gen_fn(X_batch ,X_words,lam1) 
        
        updates_dis = lasagne.updates.adam(cost_disc, params_dis , \
                                           learning_rate = lr, beta1=0.5, beta2 = 0.999, epsilon=1e-08)  #beta1=0.9
        
        updates_gen = lasagne.updates.adam(cost_gen, params_gen , \
                                           learning_rate = lr, beta1=0.5, beta2 = 0.999, epsilon=1e-08)        
        
        
        discriminator_update = theano.function([i,k,theano.In(lr,value=self.epsilon_dis)],\
                outputs=[cost_disc], updates=updates_dis,\
                givens={X_batch:train_chunk[i*self.batch_sz:(i+1)*self.batch_sz,:,:,:],
                        X_words:train_caption_chunk[i*self.batch_sz:(i+1)*self.batch_sz,\
                                                  k:(k+1),:,:]}) #.reshape((self.batch_sz,self.X_words_lgth,self.sim_dict_lgth))
        
        
        generator_update = theano.function([i,k,theano.In(lr,value=self.epsilon_gen)],\
                outputs=[cost_gen], updates=updates_gen,\
                givens={X_batch:train_chunk[i*self.batch_sz:(i+1)*self.batch_sz,:,:,:],
                        X_words:train_caption_chunk[i*self.batch_sz:(i+1)*self.batch_sz,\
                                                  k:(k+1),:,:]}) #.reshape((self.batch_sz,self.X_words_lgth,self.sim_dict_lgth))

        
        
        get_valid_cost   = theano.function([i,k], \
                                           outputs=[cost_disc,cost_gen,gen_images_gen,Orig_images],\
                givens={X_batch:valid_chunk[i*self.batch_sz:(i+1)*self.batch_sz,:,:,:],
                        X_words:val_caption_chunk[i*self.batch_sz:(i+1)*self.batch_sz,\
                                                  k:(k+1),:,:]}) #.reshape((self.batch_sz,self.X_words_lgth,self.sim_dict_lgth))

        
        return discriminator_update,generator_update , get_valid_cost 
        
    
    def cost_gen_images_fn(self,X_ORIGINAL,X_SENTENCE,lam1):
        """compute cost of the discriminator"""
        
        
        '''Generate images '''
        X_CONTEXT = X_ORIGINAL*self.Zero_middle_shd  
        
        
        self.layer_gen_save = self.gen_network.generate_image(X_CONTEXT,X_SENTENCE)
        
        gen_images = lasagne.layers.get_output(self.layer_gen_save,deterministic=True)
        
        gen_images_ctxt  = X_CONTEXT + (gen_images*self.Zero_contour_shd)
        
        
        return gen_images_ctxt.dimshuffle(0,2,3,1) #pour avoir des images : (64,64,3)

    
    def test_gan(self, test_chunk, test_caption_chunk,lam1=0.000001):
    
        """
        
        """
        X_batch = T.tensor4(name='X_batch',dtype=theano.config.floatX);
        X_words = theano.tensor.tensor4(name='X_words',dtype='float32');
        i = T.scalar('i',dtype='int32');
        #k = T.scalar('k',dtype='int32');        
        
        """compute cost of the discriminator"""
        
        gen_images_ctxt = self.cost_gen_images_fn(X_batch,X_words,lam1) 
         
        
        get_test_images   = theano.function([i], outputs=[gen_images_ctxt],\
                                          givens={X_batch:test_chunk[i*self.batch_sz:(i+1)*self.batch_sz,:,:,:],\
                                                  X_words:test_caption_chunk[i*self.batch_sz:(i+1)*self.batch_sz]})
        #.reshape((self.batch_sz,self.X_words_lgth,self.sim_dict_lgth))                                      

        return get_test_images 
  
    

