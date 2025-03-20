"""
Run the neural network training on MNIST data
"""

from network import Network
import mnist_loader

def main():
    # Load the MNIST data
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("Data loaded successfully!")

    # Create the network with 784 inputs, 30 hidden neurons, and 10 outputs
    print("\nCreating neural network [784, 30, 10]...")
    net = Network([784, 30, 10])
    print("Network created successfully!")

    # Train the network
    print("\nStarting training...")
    print("Parameters:")
    print("- Epochs: 50")
    print("- Mini-batch size: 50")
    print("- Learning rate (eta): 3.0")
    print("\nTraining progress (this may take a few minutes):")
    net.SGD(training_data, epochs=50, mini_batch_size=50, eta=3.0, test_data=test_data)
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 