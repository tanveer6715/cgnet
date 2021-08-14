import tensorflow as tf

loss_object =tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits = True,
    reduction=tf.keras.losses.Reduction.NONE
)



#@tf.function
def compute_loss(lables, predictions, class_weight): 
    """Compute loss 

    Args : 
        lables (tf.Tensor)
        predictions (tf.Tensor)
        class_weight (list)

    Return : 
        loss (tf.float)
    
    """
    loss = loss_object(lables, predictions)
    weight_map = tf.ones_like(loss)

    for idx in range(19):
        # for indexing not_equal has to be used...
        class_idx_map = tf.math.not_equal(tf.squeeze(lables), idx)
        
        weight_map = tf.where(class_idx_map, weight_map, class_weight[idx])

    loss = tf.math.multiply(loss, weight_map)

    loss = tf.reduce_mean(loss)
    

    return loss
