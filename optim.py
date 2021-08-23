import tensorflow as tf 

def load_optimizer(init_learn_rate, end_learn_rate, power): 
    """ Set optimizer for training 

    Args : 
        init_learn_rate (float) : initial learning rate 
        end_learn_rate (float) : end learning rate 
        power (float) : power 

    Returns : 
        optimizer (tf.optimizer object) : optimizer 
    
    """
    
    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(init_learn_rate, 60000, 
                                                                end_learning_rate=end_learn_rate, power=power,
                                                                cycle=False, name=None)


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08)

    return optimizer 