import numpy as np
import tensorflow as tf
import pickle
from gan_model import build_generator, build_discriminator, compile_gan
import matplotlib.pyplot as plt
import os
import pandas as pd

def train_gan(
    X_train, 
    epochs=1000, 
    batch_size=32, 
    noise_dim=10, 
    save_interval=100
):
    """
    Trains the GAN model.

    Parameters:
    - X_train: Training features.
    - epochs: Number of training epochs.
    - batch_size: Size of each training batch.
    - noise_dim: Dimension of the noise vector for the Generator.
    - save_interval: Interval at which to save generated samples and model weights.
    """
    # Normalize the data
    X_train_norm = (X_train - X_train.mean()) / X_train.std()
    real_data = X_train_norm.values
    
    # Dimensions
    data_dim = real_data.shape[1]
    
    # Build and compile models
    generator = build_generator(input_dim=noise_dim)
    discriminator = build_discriminator(input_dim=data_dim)  # Ensure the input matches the real data shape
    gan = compile_gan(generator, discriminator)
    
    # Labels
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))
    
    # For tracking loss
    d_losses, g_losses = [], []
    
    for epoch in range(1, epochs + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select a random batch of real data
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_samples = real_data[idx]
        
        # Generate fake data
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generated_samples = generator.predict(noise)
        
        # Reshape for Discriminator
        real_samples = real_samples.reshape(batch_size, data_dim)
        generated_samples = generated_samples.reshape(batch_size, data_dim)
        
        # Train on real and fake data
        d_loss_real = discriminator.train_on_batch(real_samples, real_label)
        d_loss_fake = discriminator.train_on_batch(generated_samples, fake_label)
        
        # Unpack d_loss (which likely contains both loss and accuracy)
        d_loss_real_value, d_loss_real_acc = d_loss_real[0], d_loss_real[1]
        d_loss_fake_value, d_loss_fake_acc = d_loss_fake[0], d_loss_fake[1]
        
        # Average loss for real and fake samples
        d_loss_value = 0.5 * (d_loss_real_value + d_loss_fake_value)
        d_loss_acc = 0.5 * (d_loss_real_acc + d_loss_fake_acc)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, real_label)  # We want the generator to fool the discriminator
        
        # Unpack g_loss if it's a list or tuple
        if isinstance(g_loss, (list, tuple)):
            g_loss_value = g_loss[0]
        else:
            g_loss_value = g_loss
        
        # Record the losses
        d_losses.append(d_loss_value)
        g_losses.append(g_loss_value)
        
        # Print progress
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss_value:.4f}, D Acc.: {d_loss_acc*100:.2f}% | G Loss: {g_loss_value:.4f}")
        
        # Save models at intervals
        if epoch % save_interval == 0 or epoch == epochs:
            save_models(generator, discriminator, epoch)
            plot_losses(d_losses, g_losses, epoch)


    
    # Final save
    save_models(generator, discriminator, 'final')
    plot_losses(d_losses, g_losses, epochs)
    print("Training complete and models saved.")

def save_models(generator, discriminator, epoch):
    """
    Saves the generator and discriminator models using Pickle.

    Parameters:
    - generator: The Generator model.
    - discriminator: The Discriminator model.
    - epoch: The current epoch number or 'final'.
    """
    model_dir = f'models/epoch_{epoch}'
    os.makedirs(model_dir, exist_ok=True)

    # Save models using Pickle
    with open(f"{model_dir}/generator.pkl", "wb") as f:
        pickle.dump(generator, f)

    with open(f"{model_dir}/discriminator.pkl", "wb") as f:
        pickle.dump(discriminator, f)

    print(f"Models saved for epoch {epoch}.")

def plot_losses(d_losses, g_losses, epoch):
    """
    Plots and saves the Discriminator and Generator losses.

    Parameters:
    - d_losses: List of Discriminator losses.
    - g_losses: List of Generator losses.
    - epoch: Current epoch number.
    """
    plt.figure(figsize=(10,5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title(f'Losses at Epoch {epoch}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'losses/epoch_{epoch}.png')
    plt.close()

if __name__ == "__main__":
    # Path to the CSV files
    X_train_path = './Dataset/X_train.csv'
    X_test_path = './Dataset/X_test.csv'
    
    # Load preprocessed data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    
    # Create directories for saving models and losses
    os.makedirs('models', exist_ok=True)
    os.makedirs('losses', exist_ok=True)
    
    # Train the GAN
    train_gan(X_train, epochs=1000, batch_size=32, noise_dim=10, save_interval=200)
