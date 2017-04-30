import lasagne

from conv_upconv_layer import*

from captions_context_PRJ import*

class GENERATOR_NET(object):
    def __init__(self, batch_size=4, num_filters=[8,16,32,64], sim_dict_lgth=10, X_words_lgth=6):
        
        """Parameter Initialization for Batch Norm"""
        self.batch_size = batch_size
        self.num_filters=num_filters
        self.sim_dict_lgth = sim_dict_lgth
        self.X_words_lgth=X_words_lgth

        self.L_upcon_1 = Conv_Upconv_layer()
        self.L_upcon_2 = Conv_Upconv_layer()
        self.L_upcon_3 = Conv_Upconv_layer()
        self.L_upcon_4 = Conv_Upconv_layer()
        
    
    def generate_image(self, X , testF=False, activation_type='relu'):
        
        
        l_in_g = lasagne.layers.InputLayer ( shape = (self.batch_size , self.num_filters[3], 2, 2), input_var = X )
               
        upconv_netw_1= self.L_upcon_1.upconv( l_in_g , self.num_filters[2], activation_type='relu', testF=False)
        #upconv_netw_1 =lasagne.layers.DropoutLayer(upconv_netw_1 , p=0.5)

        upconv_netw_2= self.L_upcon_2.upconv( upconv_netw_1 , self.num_filters[1], activation_type='relu', testF=False)
        #upconv_netw_2 =lasagne.layers.DropoutLayer(upconv_netw_2 , p=0.5)
        
        upconv_netw_3= self.L_upcon_3.upconv( upconv_netw_2 , self.num_filters[0], activation_type='relu', testF=False)
        #upconv_netw_3 =lasagne.layers.DropoutLayer(upconv_netw_3 , p=0.5)
        
        upconv_netw_4= self.L_upcon_4.upconv_out( upconv_netw_3 , 3 , activation_type='sigmoid', testF=False)        
        #upconv_netw_4 =lasagne.layers.DropoutLayer(upconv_netw_4 , p=0.5)
        
        return upconv_netw_4
    
