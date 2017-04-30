import lasagne
from CONV_DISCR_layer import*
from CONV_POOL_DISCR_layer import*
from CONV_DISCR_out_layer import*

class DISCRIMINATOR_NET():
    theano.config.warn_float64='raise'
    def __init__ (self,batch_size=4, num_filters=[2,4,8,16,32,32,32,32,1]):
            
            self.batch_size = batch_size

            self.L1=Conv_discr_layer(batch_size=self.batch_size,num_filters=num_filters[0],filter_channels=3)
            self.L1_pool=Conv_pool_discr_layer(batch_size=self.batch_size,num_filters=num_filters[1],filter_channels=num_filters[0])
            self.L2_pool=Conv_pool_discr_layer(batch_size=self.batch_size,num_filters=num_filters[3],filter_channels=num_filters[1])
            self.L3_pool=Conv_pool_discr_layer(batch_size=self.batch_size,num_filters=num_filters[5],filter_channels=num_filters[3])
            self.L4_pool=Conv_pool_discr_layer(batch_size=self.batch_size,num_filters=num_filters[7],filter_channels=num_filters[5])
            self.L5_OUT=Conv_discr_out_layer(batch_size=self.batch_size,num_filters=num_filters[8],filter_channels=num_filters[7]) 
            
                
    def propagate(self, l_in_d): 
        
        """
        l_in_d: input layer
        """
        
        D1=self.L1.conv_discr(l_in_d)
        
        D1_pool=self.L1_pool.conv_pool_discr_first(D1)        
        #D1_pool =lasagne.layers.DropoutLayer(D1_pool , p=0.5)

        D2=D1_pool
        D2_pool=self.L2_pool.conv_pool_discr(D2)
        #D2_pool =lasagne.layers.DropoutLayer(D2_pool , p=0.5)
        
        D3=D2_pool
        D3_pool=self.L3_pool.conv_pool_discr(D3)
        #D3_pool =lasagne.layers.DropoutLayer(D3_pool , p=0.5)
        
        D4=D3_pool
        D4_pool=self.L4_pool.conv_pool_discr(D4)
        #D4_pool =lasagne.layers.DropoutLayer(D4_pool , p=0.5)        
        
        D5_OUT=self.L5_OUT.conv_discr_out(D4_pool)
        #D5_OUT =lasagne.layers.DropoutLayer(D5_OUT , p=0.5)
               
        return D5_OUT 