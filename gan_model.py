from tensorflow.keras import layers, models

def build_generator(input_dim):
    """
    Builds the generator model.

    Parameters:
    - input_dim: Dimension of the noise vector.

    Returns:
    - Generator model.
    """
    model = models.Sequential()
    
    # Add layers to the generator to output the same shape as the real data
    model.add(layers.Dense(256, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    
    # Output layer to match the real data shape (assuming 1D data for now)
    model.add(layers.Dense(1, activation='tanh'))
    
    return model

def build_generator(input_dim):
    """
    Builds the generator model.

    Parameters:
    - input_dim: Dimension of the noise vector.

    Returns:
    - Generator model.
    """
    model = models.Sequential()
    
    # Add layers to the generator to output the same shape as the real data (8 features)
    model.add(layers.Dense(256, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    
    # Output 8 features to match the shape of the real data
    model.add(layers.Dense(8, activation='linear'))  # Output shape is (batch_size, 8)
    
    return model
def compile_gan(generator, discriminator):
    """
    Compiles the GAN by combining the generator and discriminator.

    Parameters:
    - generator: The generator model.
    - discriminator: The discriminator model.

    Returns:
    - Compiled GAN model.
    """
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Freeze discriminator's weights when training the generator
    discriminator.trainable = False

    # Build the GAN model (generator + discriminator)
    gan_input = layers.Input(shape=(generator.input_shape[1],))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)

    gan_model = models.Model(gan_input, gan_output)
    gan_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return gan_model

def build_discriminator(input_dim):
    """
    Builds the discriminator model.

    Parameters:
    - input_dim: Dimension of the input data (real or generated).

    Returns:
    - Discriminator model.
    """
    model = models.Sequential()
    
    # Input to the discriminator should match the real data shape
    model.add(layers.Dense(512, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output is a single value (real/fake)
    
    return model