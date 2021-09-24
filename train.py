import argparse
import os 

import tensorflow as tf
import numpy as np

from datasets.cityscapes import CityscapesDatset
from datasets.concrete_damage_as_cityscapes import Concrete_Damage_Dataset_as_Cityscapes
from model import CGNet
from pipeline import batch_generator
from utils import load_config
from loss import compute_loss
from optim import load_optimizer
import time

def distributed_train_step(dist_inputs, mirrored_strategy):
    """
    """
    per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    return mirrored_strategy.reduce("MEAN", per_replica_losses,
                        axis=None)


@tf.function
def train_step(inputs):
    """Train step 

    Args : 
        inputs (tuple) : includes images (tf.tensor) and labels (tf.tensor)
        train_loss (global, tf.keras.metric)
        train_accuracy (global, tf.keras.metric)
        train_iou (global, tf.keras.metric)
        optimizer (gloabl, tf.optimizer)

    Return 
        loss 
    
    """

    images, labels = inputs
    
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = compute_loss(labels, predictions, class_weight)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    argmax_predictions = tf.math.argmax(predictions, 3)
    train_iou.update_state(labels, argmax_predictions)

    return loss


def train_model(model, train_dataset, optimizer,
                train_loss, train_accuracy, train_iou,
                batch_size, class_weight, epochs, 
                num_steps, log_template, model_save_to ):

    """Train model on a single gpu
    Args : 
        model (tf.model) 
        train_dataset (generator)
        optimizer (tf.optimizer)
        train_loss (tf.keras.metric)
        train_accuracy (tf.keras.metric) 
        train_iou (tf.keras.metric) 
        batch_size (int or float) 
        class_weight (list) 
        epochs (int or float) 
        num_steps (int or float) 
        log_template (string) 
        model_save_to (string)

    Returns : 
        None 
    """

    for epoch in range(1, epochs):

        train_dataset_generator = batch_generator(train_dataset, batch_size)

        for step, batch in enumerate(train_dataset_generator):
        
            train_step(batch)
            
            if step % 5 == 1 : 
                print(log_template.format(epoch,
                                        epochs,
                                        step+1,
                                        num_steps,
                                        train_loss.result(),
                                        train_accuracy.result()*100,
                                        train_iou.result()*100
                                        ))
                                        
        if epoch % 5 == 0 :
            model.save_weights(os.path.join(model_save_to, 'epoch_{}.h5'.format(epoch)))
        if epoch % 300 == 0 :
            model.save(os.path.join(model_save_to))#,save_traces = False)

def dist_train(model, train_dataset, optimizer,
                train_loss, train_accuracy, train_iou,
                batch_size, class_weight, epochs, 
                num_steps, log_template, model_save_to):

    def gen():  
        """
        """
        for images, labels in train_dataset_generator:
            images = tf.squeeze(images)
            labels = tf.squeeze(labels, axis = 0)
            yield images, labels

    mirrored_strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

   #with mirrored_strategy.scope():
    #     dist_model = model
    #     dist_optimizer = optimizer

    #     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #                 from_logits = True,
    #                 reduction=tf.keras.losses.Reduction.NONE)


        # model_save_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.20 concrete/'
        # model.build((4,680,680,3))
        # keras.models.load_model(model_save_path)
    train_dataset_generator = batch_generator(train_dataset, 1, repeat= epochs)

    tf_train_dataset_generator = tf.data.Dataset.from_generator(gen, (tf.float32,  tf.uint8), 
                                                    ((680, 680, 3), (680, 680, 1)))

    tf_train_dataset_generator = tf_train_dataset_generator.batch(batch_size)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_train_dataset_generator = tf_train_dataset_generator.with_options(options)
    dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(tf_train_dataset_generator)

    train_dataset_iterator = iter(dist_train_dataset)

    for epoch in range(1, epochs + 1):
        
        for step in range(1, num_steps+1):
            distributed_train_step(next(train_dataset_iterator), mirrored_strategy)
        
            if step % 5 == 1 : 
                print(log_template.format(epoch,
                                        epochs,
                                        step+1,
                                        num_steps,
                                        train_loss.result(),
                                        train_accuracy.result()*100,
                                        train_iou.result()*100
                                        ))

        if epoch % 5 == 0 :
            model.save_weights(os.path.join(model_save_to, 'epoch_{}.h5'.format(epoch)))
        if epoch % 300 == 0 :
            model.save(os.path.join(model_save_to))#,save_traces = False)
if __name__ == "__main__" : 

    
    parser = argparse.ArgumentParser("Please Set Training Configuration File")
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    config = load_config(args.config_path)


    data_dir = config.get('DATA_DIR', None)
    resume_from = config.get('RESUME_FROM', None)
    model_save_to = config.get('MODEL_SAVE_TO', None)

    dataset = config.get('DATASET', None)
    img_width = config.get('IMG_WIDTH', None)
    img_height = config.get('IMG_HEIGHT', None)
    num_classes = config.get('NUM_CLASSES', None)
    ignore_calss = config.get('IGNORE_CLASS', None)

    batch_size = config.get('BATCH_SIZE', None)
    epochs = config.get('EPOCHS', None)
    num_gpu = config.get('NUM_GPU', None)

    num_m_blocks = config.get('NUM_M_BLOCKS', None)
    num_n_blocks = config.get('NUM_N_BLOCKS', None)

    init_learn_rate = config.get('INIT_LEARN_RATE', None)
    end_learn_rate = config.get('END_LEARN_RATE',  None)
    power = config.get('POWER', None)

    class_weight = config.get('CLASS_WEIGHT', None)

    log_template = 'Epoch: {}/{}, steps:{}/{}, Loss: {:2f}, Accuracy: {:2f}, MeanIoU: {:2f}'


    model = CGNet(num_classes = num_classes, M= num_m_blocks, N=num_n_blocks)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_iou = tf.keras.metrics.MeanIoU(num_classes= num_classes, name='train_miou')


    if dataset == 'Cityscapes':
        train_dataset = CityscapesDatset(data_dir)

    elif dataset == 'Concrete_Damage_Cityscapes':
        train_dataset = Concrete_Damage_Dataset_as_Cityscapes(data_dir)

    num_steps = len(train_dataset)//batch_size

    optimizer = load_optimizer(init_learn_rate, end_learn_rate, power)

    if num_gpu == 1 :
        train_model(model, train_dataset, optimizer,
                train_loss, train_accuracy, train_iou,
                batch_size, class_weight, epochs, 
                num_steps, log_template, model_save_to,resume_from )

    elif num_gpu >= 2: 
        dist_train(model, train_dataset, optimizer,
                train_loss, train_accuracy, train_iou,
                batch_size, class_weight, epochs, 
                num_steps, log_template, model_save_to )
