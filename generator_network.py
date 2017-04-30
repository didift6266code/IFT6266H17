import lasagne
#from conv_network import *
#from upconv_network import *

from conv_upconv_layer import*

from captions_context_PRJ import*

class GENERATOR_NET(object):
    
    
    def __init__(self, batch_size=4, num_filters=[8,16,32,64], sim_dict_lgth=10, X_words_lgth=6):
        
        """Parameter Initialization for Batch Norm"""
        self.batch_size = batch_size
        self.num_filters=num_filters
        self.sim_dict_lgth = sim_dict_lgth
        self.X_words_lgth=X_words_lgth
        
        self.L_con_1 = Conv_Upconv_layer(batch_size=batch_size, num_filters=num_filters[0],filter_channels=3)
        self.L_con_2 = Conv_Upconv_layer(batch_size=batch_size, num_filters=num_filters[1], filter_channels=num_filters[0])
        self.L_con_3 = Conv_Upconv_layer(batch_size=batch_size, num_filters=num_filters[2], filter_channels=num_filters[1])
        self.L_con_4 = Conv_Upconv_layer(batch_size=batch_size, num_filters=num_filters[3], filter_channels=num_filters[2])
    
        self.L_upcon_1 = Conv_Upconv_layer()
        self.L_upcon_2 = Conv_Upconv_layer()
        self.L_upcon_3 = Conv_Upconv_layer()
        self.L_upcon_4 = Conv_Upconv_layer()
        
        self.CAP_CTXT = CAPTIONS_CONTEXT(batch_size=self.batch_size,sim_dict_lgth=self.sim_dict_lgth, X_words_lgth=self.X_words_lgth)    
    
    def generate_image(self, X , X_words , testF=False, activation_type='relu'):
        
        l_in_g_0 = lasagne.layers.InputLayer ( shape = (self.batch_size , 3, 64,64), input_var = X )
        
        l_in_g_1 = lasagne.layers.Conv2DLayer(l_in_g_0, num_filters=3, filter_size=(5,5), stride=(1, 1), pad='same', nonlinearity=lasagne.nonlinearities.sigmoid)
        
        l_in_g = lasagne.layers.batch_norm(l_in_g_1)
        
        conv_netw_1, conv_netw_1_tide= self.L_con_1.conv( l_in_g, activation_type='relu', testF=False)
        #conv_netw_1 =lasagne.layers.DropoutLayer(conv_netw_1 , p=0.5)
        
        conv_netw_2, conv_netw_2_tide= self.L_con_2.conv( conv_netw_1, activation_type='relu', testF=False)
        #conv_netw_2 =lasagne.layers.DropoutLayer(conv_netw_2 , p=0.5)
        
        conv_netw_3, conv_netw_3_tide= self.L_con_3.conv( conv_netw_2, activation_type='relu', testF=False)
        #conv_netw_3 =lasagne.layers.DropoutLayer(conv_netw_3 , p=0.5)
        
        conv_netw_4, conv_netw_4_tide= self.L_con_4.conv( conv_netw_3, activation_type='relu', testF=False)
        
        #conv_netw_4 =lasagne.layers.DropoutLayer(conv_netw_4 , p=0.5)
        
        
        layer_align = self.CAP_CTXT.add_caption_context(X_words)
        #layer_align =lasagne.layers.DropoutLayer(layer_align , p=0.5)
        
        conv_netw_4_concat=lasagne.layers.ConcatLayer((conv_netw_4,layer_align), axis=1, cropping=None)    
       
    
        upconv_netw_1= self.L_upcon_1.upconv_concat( conv_netw_4_concat , conv_netw_4_tide, activation_type='relu', testF=False)
        #upconv_netw_1 =lasagne.layers.DropoutLayer(upconv_netw_1 , p=0.5)
        
        upconv_netw_2= self.L_upcon_2.upconv( upconv_netw_1 , conv_netw_3_tide, activation_type='relu', testF=False)
        #upconv_netw_2 =lasagne.layers.DropoutLayer(upconv_netw_2 , p=0.5)
        
        upconv_netw_3= self.L_upcon_3.upconv( upconv_netw_2 , conv_netw_2_tide, activation_type='relu', testF=False)
        #upconv_netw_3 =lasagne.layers.DropoutLayer(upconv_netw_3 , p=0.5)
        
        upconv_netw_4= self.L_upcon_4.upconv_out( upconv_netw_3 , conv_netw_1_tide, activation_type='sigmoid', testF=False)        
        #upconv_netw_4 =lasagne.layers.DropoutLayer(upconv_netw_4 , p=0.5)
        
        return upconv_netw_4
    
