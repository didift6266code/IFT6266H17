import theano 
import numpy as np
import scipy as sp
import lasagne
from lasagne.regularization import regularize_network_params, l2 
from generator_network import*
from discriminator_network import *
theano.config.floatX = 'float32'

class MODEL_OPTIMIZER():

    def __init__(self,batch_sz, epsilon_dis=0.0000002,epsilon_gen=0.0000002,num_filters_g=[64,128,256,512],\
                num_filters_d=[32,64,128,256,512,512,512,512,1], sim_dict_lgth=10, X_words_lgth=62):
        """Initializes discriminator and generator and optimise their loss functions"""
        
        self.epsilon_dis = epsilon_dis
        self.epsilon_gen = epsilon_gen
        self.sim_dict_lgth= sim_dict_lgth
        self.X_words_lgth=X_words_lgth
        self.batch_sz = batch_sz
        self.dis_network = DISCRIMINATOR_NET(self.batch_sz , num_filters_d) 
        self.gen_network = GENERATOR_NET(self.batch_sz , num_filters_g, sim_dict_lgth=self.sim_dict_lgth,\
                                         X_words_lgth=self.X_words_lgth)
        
        Zero_middle = np.ones((self.batch_sz,3,64,64),dtype = np.float32)
        Zero_middle[:,:,16:48,16:48] = 0.0
        
        Zero_complete = np.zeros((self.batch_sz,3,64,64),dtype = np.float32)
        Zero_contour = np.zeros((self.batch_sz,3,64,64),dtype = np.float32)
        Zero_contour[:,:,16:48,16:48] = 1.0
        
        self.Zero_middle_shd = theano.shared(Zero_middle, name='Zero_middle_shd')
        self.Zero_contour_shd = theano.shared(Zero_contour, name='Zero_contour_shd')
        self.Zero_complete_shd = theano.shared(Zero_contour, name='Zero_complete_shd')
        
    def cost_dis_fn(self,X_ORIGINAL,X_SENTENCE,lam1):
        """compute cost of the discriminator"""

        '''Discriminate of the original image '''
        target1  = T.alloc(.9, self.batch_sz)
        l_in_d = lasagne.layers.InputLayer ( shape = (self.batch_sz , 3, 64,64), input_var = X_ORIGINAL )
        p_y__x1  = lasagne.layers.get_output(self.dis_network.propagate(l_in_d))
        p_y__x1  = p_y__x1.flatten()
        

        X_CONTEXT           = theano.tensor.set_subtensor(X_ORIGINAL[:,:,16:48,16:48], 0.0)

        layer_gen_out        = self.gen_network.generate_image(X_ORIGINAL,X_SENTENCE)#(X_CONTEXT,X_SENTENCE)
        self.layer_gen_save  = layer_gen_out
        
        layer_dis_out        = self.dis_network.propagate(self.layer_gen_save)
        
        '''Discriminate of the original image context with the generated image middle in its middle'''
        target0             = T.alloc(0., self.batch_sz)   
        gen_images          = lasagne.layers.get_output(self.layer_gen_save)
        gen_images_context  = theano.tensor.set_subtensor(X_CONTEXT[:,:,16:48,16:48] , gen_images[:,:,16:48,16:48] )

        l_in_d_0            = lasagne.layers.InputLayer ( shape = (self.batch_sz , 3, 64,64), input_var = gen_images_context )
        layer_dis_out_0     = self.dis_network.propagate(l_in_d_0)
        '''les parametres du discriminator seuls'''
        params_dis          = lasagne.layers.get_all_params(layer_dis_out_0,trainable=True)
        p_y__x0             = lasagne.layers.get_output(layer_dis_out_0)
        p_y__x0             = p_y__x0.flatten() 
        
        
        
        lam_w_decay= lam1*regularize_network_params(layer=layer_dis_out_0, penalty=l2)
        
        

        return (T.mean(T.nnet.binary_crossentropy(p_y__x1, target1)) \
                       + T.mean(T.nnet.binary_crossentropy(p_y__x0, target0)))+lam_w_decay,\
               params_dis,gen_images_context.dimshuffle(0,2,3,1)




    def cost_gen_fn(self,X_ORIGINAL,X_SENTENCE,lam1):
        """compute cost of the discriminator"""
        
        
        '''Discriminate of the generated image '''

        X_CONTEXT           = theano.tensor.set_subtensor(X_ORIGINAL[:,:,16:48,16:48], 0.0)
        
        layer_gen_out        = self.gen_network.generate_image(X_ORIGINAL,X_SENTENCE)#(X_CONTEXT,X_SENTENCE)
        params_gen           = lasagne.layers.get_all_params(layer_gen_out,trainable=True)
        
        '''Discriminate of the original image context with the generated image middle in its middle'''
        target1             = T.alloc(.9, self.batch_sz)   
        gen_images          = lasagne.layers.get_output(layer_gen_out)
        
        gen_images_context  = theano.tensor.set_subtensor(X_CONTEXT[:,:,16:48,16:48] , gen_images[:,:,16:48,16:48] )
        
        l_in_d_0            = lasagne.layers.InputLayer ( shape = (self.batch_sz , 3, 64,64), input_var = gen_images_context )
        p_y__x1             = lasagne.layers.get_output(self.dis_network.propagate(l_in_d_0))
        p_y__x1             = p_y__x1.flatten() 
        
        lam_w_decay= lam1*regularize_network_params(layer=layer_gen_out, penalty=l2)
        
        
        return (T.mean(T.nnet.binary_crossentropy(p_y__x1, target1)) )+lam_w_decay,\
                       params_gen,gen_images_context.dimshuffle(0,2,3,1), X_ORIGINAL.dimshuffle(0,2,3,1)
        
        
            

        

    def optimize_gan(self, train_chunk, valid_chunk, train_caption_chunk,val_caption_chunk,\
                     lam1=0.000001):
    
        """
        
        """

        lr = theano.tensor.fscalar('lr');
        #drpout = theano.tensor.bscalar('drpout');
        X_batch = theano.tensor.tensor4(name='X_batch',dtype=theano.config.floatX);
        X_words = theano.tensor.tensor4(name='X_words',dtype='float32');
        i = theano.tensor.scalar('i',dtype='int32');
        k = theano.tensor.scalar('k',dtype='int32');
        
        
        """compute cost of the discriminator"""
        
        cost_disc , params_dis, gen_images = self.cost_dis_fn(X_batch,X_words,lam1) 
        
        cost_gen , params_gen, gen_images_gen, Orig_images = self.cost_gen_fn(X_batch,X_words,lam1) 
        
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
  
    

