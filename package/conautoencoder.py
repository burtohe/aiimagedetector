import tensorflow as tf

class CONAutoencoder(tf.keras.Model):
    def __init__(self, input_shape):
        super(CONAutoencoder, self).__init__()
        
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def summary_alt(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'))
        

        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
            
        model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
        
        model.summary()
        
        del model
        
        
        
class CONAutoencoder2(tf.keras.Model):
    def __init__(self, input_shape):
        super(CONAutoencoder2, self).__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def summary_alt(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
            
        model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
        
        model.summary()
        
        del model
        
        
class CONAutoencoder3(tf.keras.Model):
    def __init__(self, input_shape):
        super(CONAutoencoder3, self).__init__()
        
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(40, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(40, (3, 3), activation='relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def summary_alt(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(40, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'))
        

        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
            
        model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(40, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
        
        model.summary()
        
        del model