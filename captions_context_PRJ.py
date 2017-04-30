import theano
import lasagne

class CAPTIONS_CONTEXT(object):
    
    def __init__(self, batch_size=4,sim_dict_lgth=10, X_words_lgth=6):
        
        """Parameter Initialization for Batch Norm"""
        self.batch_size = batch_size
        self.sim_dict_lgth  = sim_dict_lgth
        self.X_words_lgth = X_words_lgth
        self.W_in=lasagne.init.Normal(0.1)
        self.W_hid=lasagne.init.Normal(0.1)
        self.W_cell=lasagne.init.Normal(0.1)
        
        
    def add_caption_context(self,X_words):
        
        self.l_in_lstm = lasagne.layers.InputLayer ( shape = (self.batch_size ,1, self.X_words_lgth,  self.sim_dict_lgth),\
                                     input_var = X_words ) 
        
        
        self.lstm_layer_f=lasagne.layers.LSTMLayer(self.l_in_lstm, num_units=self.X_words_lgth, 
                         ingate=lasagne.layers.Gate(W_in=self.W_in, W_hid=self.W_hid, W_cell=self.W_cell), 
                         forgetgate=lasagne.layers.Gate(W_in=self.W_in, W_hid=self.W_hid, W_cell=self.W_cell), 
                         cell=lasagne.layers.Gate(W_in=self.W_in, W_hid=self.W_hid, W_cell=None,
                                                  nonlinearity=lasagne.nonlinearities.tanh), 
                         outgate=lasagne.layers.Gate(W_in= self.W_in, W_hid=self.W_hid, W_cell=self.W_cell), 
                         nonlinearity=lasagne.nonlinearities.tanh, 
                         cell_init=lasagne.init.Constant(0.),
                         hid_init=lasagne.init.Constant(0.),
                         backwards=False, learn_init=False, peepholes=True, gradient_steps=-1,
                         grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None,
                         only_return_final=True)
        self.lstm_layer_f=lasagne.layers.DimshuffleLayer(self.lstm_layer_f, (0,1,'x'))
        
        self.lstm_layer_b=lasagne.layers.LSTMLayer(self.l_in_lstm, num_units=self.X_words_lgth, 
                         ingate=lasagne.layers.Gate(W_in=self.W_in, W_hid=self.W_hid, W_cell=self.W_cell), 
                         forgetgate=lasagne.layers.Gate(W_in=self.W_in, W_hid=self.W_hid, W_cell=self.W_cell), 
                         cell=lasagne.layers.Gate(W_in= self.W_in, W_hid=self.W_hid, W_cell=None, 
                                                  nonlinearity=lasagne.nonlinearities.tanh), 
                         outgate=lasagne.layers.Gate(W_in=self.W_in, W_hid=self.W_hid, W_cell=self.W_cell), 
                         nonlinearity=lasagne.nonlinearities.tanh, 
                         cell_init=lasagne.init.Constant(0.),
                         hid_init=lasagne.init.Constant(0.),
                         backwards=True, learn_init=False, peepholes=True, gradient_steps=-1,
                         grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None,
                         only_return_final=True)
        self.lstm_layer_b=lasagne.layers.DimshuffleLayer(self.lstm_layer_b, (0,1,'x'))
        
        self.lstm_concat=lasagne.layers.ConcatLayer((self.lstm_layer_f, self.lstm_layer_b), axis=2, cropping=None)
        
        
        self.l_align_final=lasagne.layers.DenseLayer(self.lstm_concat, num_units=3*16, 
                                                        W=lasagne.init.GlorotUniform(),
                                                        b=None,
                                                        nonlinearity=None,
                                                        num_leading_axes=1)

        self.l_align_final = lasagne.layers.ReshapeLayer(self.l_align_final, (self.batch_size,3,4,4)) 
        
        return self.l_align_final
        
