from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv3D, Deconve3D 
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding3D
import matplotlib.pyplot as plt
import numpy as np

        
class GAN():
    def __init__(self):
        self.image_rows = 64
        self.image_cols = 64
        self.image_depth = 64
        self.channels = 3
         self.kernel_size =(1,1,1)
        self.stride = (2,2,2)
        self.image_shape = (self.image_rows, self.image_cols, self.image_depth, self.channels, self.kernel_size, self.stride)
        self.latent_dim = 200
        optimizer = Adam(0.001, 0.5)    
         self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='optimizer',
        metrics=["accuracy"]
        
         self.generator = self.build_generator()
     # The generator takes noise as input and generates images     
          z = Input(shape=(self.latent_dim))
        image = self.generator(z)
 # For the combined model we will only train the generator        
 self.discriminator.trainable = False 
 
 # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(image)
        
 # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
data=np.load("modelnet10.npz", allow pickle=True)
        train_voxel=data["train_voxel"]
        test_voxel=data["test_voxel"]
        train_labels=data["train_labels"]
        test_labels=data["test_labels"]
        
def build_generator(self):

        model = Sequential()
        model.add(Deconv3D(filters=512, kernel_size=self.kernel_size,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=256, kernel_size=self.kernel_size,
                      strides=self.strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=128, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=64, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=1, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        noise = Input(shape=(1, 1, 1, self.latent_dim))
        image = model(noise)

        return Model(inputs=noise, outputs=image)
    model.summary()
    
 def build_discriminator(self):

         model = Sequential()
        model.add(Conv3D(filters=64, kernel_size=self.kernel_size,
                    strides=self.strides, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=128, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=256, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=512, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=1, kernel_size=self.kernel_size,
                         strides=(1, 1, 1), kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        image = Input(shape=(self.im_dim, self.im_dim, self.im_dim, 1))
        validity = model(image)

        return Model(inputs=image, outputs=valid)
    model.summary()                
        
 def plot_generated_images(self,epoch, generator):
        
        noise = np.random.normal(0, 1, (self.sample_images, self.latent_dim))
        generated_images = self.generator.predict(noise)
        plt.figure(figsize=(10,10))
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0],dim[1],i+1)
            plt.imshow(generated_images[i], interpolation='nearest', colormap=(0,0,1))
            plt.savefig(f'Gan_generated_images_epoch_{epoch.png}')
      
        
     def train(self, epoch=40, batch_size=30, sample_interval=10):
        data=np.load("modelnet10.npz", allow pickle=True)
        train_voxel=data["train_voxel"]
        test_voxel=data["test_voxel"]
        train_labels=data["train_labels"]
        test_labels=data["test_labels"]
        class_map=data["class_map"]
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(test_voxel, test_labels, epoch, validation_split=0.2, shuffle=True)
        model.test_on_batch(test_voxel, test_labels)
     
      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1)) 
      for epoch in range(epochs):  
       #  Train Discriminator    
       x = np.random.randint(0, train_voxel[0], batch_size)
            images = train_voxel[x]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
    generated_images = self.generator.predict(noise) 
    
    # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgages, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    #  Train Generator

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
     # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
     # loss
        plt.plot(d_loss,'-')
        plt.plot(g_loss,'-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend('Generator','discriminator')
        
    
    #  "Accuracy"
        plt.plot(history.history['accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show() 
