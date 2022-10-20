import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import horovod.tensorflow.keras as hvd
import argparse
# import tensorflow_addons as tfa

#tf.random.set_seed(1233)

def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

class WGAN(keras.Model):
    """WGAN model"""
    def __init__(self, data_shape,num_cond,num_noise,config,name='wgan'):
        super(WGAN, self).__init__()
        if config is None:
            raise ValueError("Config file not given")
        self._num_cond = num_cond
        self._data_shape = data_shape
        self.config = config
        self.activation = self.config['ACT']
        #config file with ML parameters to be used during training        
        #Transformation applied to conditional inputs
        inputs_cond = Input((self._num_cond))

        self.latent_dim = num_noise
        self.d_steps = 5  #number of discriminator steps for each generator
        self.gp_weight = 10 #weight of the gradient penalty to the loss

        if len(self._data_shape) == 2:
            self.shape = (-1,1,1)
        else:
            self.shape = (-1,1,1,1,1)

        self.discriminator = self.ConvDiscriminator(inputs_cond)
        self.generator = self.ConvGenerator(inputs_cond)
        self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank
        if self.verbose:
            #print(self.discriminator.summary())
            print(self.generator.summary())

    def compile(self,d_optimizer, g_optimizer):
        super(WGAN, self).compile(experimental_run_tf_function=False)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images,cond):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        if len(self._data_shape) == 2:
            alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        else:
            alpha = tf.random.normal([batch_size, 1,1,1 ,1], 0.0, 1.0)
            
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated,cond], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        if len(self._data_shape) == 2:
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        else:
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3,4]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def generate(self,nevts,cond):
        random_latent_vectors = tf.random.normal(
            shape=(nevts, self.latent_dim)
        )
        return self.generator([random_latent_vectors,cond], training=False)
    
    def train_step(self, inputs):        
        real_images,cond = inputs
        real_images = tf.reshape(real_images,[-1]+self._data_shape)
        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                real_logits = self.discriminator([real_images,cond], training=True)
                fake_images = self.generator([random_latent_vectors,cond], training=True)
                fake_logits = self.discriminator([fake_images,cond], training=True)                

                d_cost = discriminator_loss(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images,cond)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            
            
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors,cond], training=True)
            gen_img_logits = self.discriminator([generated_images,cond], training=True)

            # Calculate the generator loss
            g_loss = generator_loss(gen_img_logits)


        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )



        

        return {"d_loss": d_loss, "g_loss": g_loss}        
                        

    def test_step(self, inputs):
        real_images,cond = inputs
        real_images = tf.reshape(real_images,[-1]+self._data_shape)
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        generated_images = self.generator([random_latent_vectors,cond], training=False)
        gen_img_logits = self.discriminator([generated_images,cond], training=False)
        real_logits = self.discriminator([real_images,cond], training=True)
        
        gp = self.gradient_penalty(batch_size, real_images, generated_images,cond)
        d_loss = discriminator_loss(real_img=real_logits, fake_img=gen_img_logits)+ gp * self.gp_weight
        g_loss = generator_loss(gen_img_logits)
        
        return {"d_loss": d_loss, "g_loss": g_loss,'gp_loss':gp}

    def conv_layer(self,input_layer,embed,hidden_size,stride=1,kernel_size=2,padding="same",activation=True,use_cond=False):
        ## Incorporate information from conditional inputs        
        if use_cond:
            cond = tf.reshape(embed,self.shape)
            layer = input_layer+cond
        else:
            layer = input_layer
            
        if len(self._data_shape) == 2:
            layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,padding=padding,
                                  strides=1,use_bias=False,activation=self.activation)(layer)
            layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,
                                  padding=padding,
                                  strides=1,use_bias=True)(layer) 
            
        else:
            layer = layers.Conv3D(hidden_size,kernel_size=kernel_size,padding=padding,
                                  strides=1,use_bias=False,activation=self.activation)(layer)
            layer = layers.Conv3D(hidden_size,kernel_size=1,
                                  padding=padding,
                                  strides=1,use_bias=True)(layer)

        # layer = layers.BatchNormalization()(layer)
        # layer = layers.Dropout(0.1)(layer)
        if activation:            
            return self.activate(layer)
        else:
            return layer
    
    def ConvDiscriminator(self,cond_embed):
        inputs = Input((self._data_shape))
        stride = self.config['STRIDE']
        conv_sizes = self.config['LAYER_SIZE']
        stride_size=self.config['STRIDE']
        kernel_size =self.config['KERNEL']
        nlayers =self.config['NLAYERS']

        layer_encoded = self.conv_layer(inputs,cond_embed,conv_sizes[0],
                                        kernel_size=kernel_size,
                                        stride=1,padding='same',use_cond=True)


        for ilayer in range(1,nlayers):
            layer_encoded = self.conv_layer(layer_encoded,cond_embed,conv_sizes[ilayer],
                                           kernel_size=kernel_size,padding='same',
                                           stride=1,
            )
                
            if len(self._data_shape) == 2:
                layer_encoded = layers.AveragePooling1D(stride_size)(layer_encoded)
            else:
                layer_encoded = layers.AveragePooling3D(stride_size)(layer_encoded)
                                
        
        self.init_shape=layer_encoded.get_shape()[1:]

        x = layers.Flatten()(layer_encoded)
        self.init_size = x.get_shape()[-1]
        # #MLP layers
        # mlp_sizes = [1024,2048,1024]

        # for mlp in mlp_sizes:
        #     x = layers.Dense(mlp)(x)
        #     x = layers.ReLU()(x)
        outputs = layers.Dense(1)(x)        

        return  keras.models.Model([inputs,cond_embed], outputs, name="discriminator")
    
    def ConvGenerator(self,cond_embed):        
        inputs = Input((self.latent_dim))
        stride = self.config['STRIDE']
        conv_sizes = self.config['LAYER_SIZE']
        stride_size=self.config['STRIDE']
        kernel_size =self.config['KERNEL']
        nlayers =self.config['NLAYERS']

        
        layer = tf.concat([inputs,cond_embed],-1)        
        layer = layers.Dense(self.init_size)(layer)
        layer = self.activate(layer)
        layer = layers.Reshape(self.init_shape)(layer)
        
        layer_decoded = self.conv_layer(layer,
                                        cond_embed,conv_sizes[nlayers-1],
                                        stride = 1,
                                        kernel_size=kernel_size,padding='same')
        for ilayer in range(nlayers-1):
            layer_decoded = self.conv_layer(layer_decoded,
                                            cond_embed,conv_sizes[nlayers-2-ilayer],
                                            stride = 1,
                                            kernel_size=kernel_size,padding='same')
            if len(self._data_shape) == 2:
                layer_decoded = layers.UpSampling1D(stride_size)(layer_decoded)                
            else:
                layer_decoded = layers.UpSampling3D(stride_size)(layer_decoded)
                    
            layer_decoded = self.conv_layer(layer_decoded,
                                            cond_embed,conv_sizes[nlayers-2-ilayer],
                                            stride = 1,
                                            kernel_size=kernel_size,padding='same')
            
        layer_decoded = self.conv_layer(layer_decoded,
                                        cond_embed,conv_sizes[0],
                                        stride = 1,
                                        kernel_size=kernel_size,padding='same')



        if len(self._data_shape) == 2:
            outputs = layers.Conv1D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(layer_decoded)
        else:
            outputs = layers.Conv3D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(layer_decoded)

        
        return  keras.models.Model([inputs,cond_embed], outputs, name="generator")

    
    def activate(self,layer):
        if self.activation == 'leaky_relu':                
            return keras.layers.LeakyReLU(0.2)(layer)
        elif self.activation == 'relu':
            return keras.activations.relu(layer)
        elif self.activation == 'swish':
            return keras.activations.swish(layer)
        else:
            raise ValueError("Activation function not supported!")   
