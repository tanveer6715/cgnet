import tensorflow as tf

loss_object =tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits = True,
    reduction=tf.keras.losses.Reduction.NONE
)



# @tf.function
# def compute_loss(lables, predictions, class_weight): 
#     """Compute loss 

#     Args : 
#         lables (tf.Tensor)
#         predictions (tf.Tensor)
#         class_weight (list)

#     Return : 
#         loss (tf.float)
    
#     """
#     loss = loss_object(lables, predictions)
#     weight_map = tf.ones_like(loss)

#     for idx in range(len(weight_map)):
#         # for indexing not_equal has to be used...
#         class_idx_map = tf.math.not_equal(tf.squeeze(lables), idx)
        
#         weight_map = tf.where(class_idx_map, weight_map, class_weight[idx])

#     loss = tf.math.multiply(loss, weight_map)

#     loss = tf.reduce_mean(loss)
    

#     return loss

def compute_loss(labels, predictions, class_weight):
    ignore_class = 255
    idx_to_ignore = labels!= ignore_class
    labels = tf.where(idx_to_ignore, labels, 0)
    
    per_example_loss = loss_object(labels, predictions)
    weight_map = tf.ones_like(per_example_loss)

    # detach labels with 255 
    for idx in range(len(weight_map)):
        # for indexing not_equal has to be used...
        class_idx_map = tf.math.not_equal(tf.squeeze(labels), idx)
        weight_map = tf.where(class_idx_map, weight_map, class_weight[idx])

    weight_map = tf.where(tf.squeeze(idx_to_ignore), weight_map, 0)
    # tf.print(weight_map)
    
    per_example_loss = tf.math.multiply(per_example_loss, weight_map)
    per_example_loss = tf.reduce_mean(tf.boolean_mask(per_example_loss,tf.math.not_equal(per_example_loss, 0)))

    return per_example_loss
