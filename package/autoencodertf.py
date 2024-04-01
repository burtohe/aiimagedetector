import tensorflow as tf

class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=shape),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim*3, activation='relu'),
      tf.keras.layers.Dense(latent_dim*2, activation='relu'),
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(latent_dim*2, activation='relu'),
      tf.keras.layers.Dense(latent_dim*3, activation='relu'),
      tf.keras.layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
      tf.keras.layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
  
  def summary_alt(self, latent_dim, shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=shape),)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(latent_dim*3, activation='relu'))
        model.add(tf.keras.layers.Dense(latent_dim*2, activation='relu'))
        
        model.add(tf.keras.layers.Dense(latent_dim, activation='relu'))
            
        model.add(tf.keras.layers.Dense(latent_dim*2, activation='relu'))
        model.add(tf.keras.layers.Dense(latent_dim*3, activation='relu'))
        
        
        model.add(tf.keras.layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'))
        model.add(tf.keras.layers.Reshape(shape))
        
        
        model.summary()
        
        del model
        