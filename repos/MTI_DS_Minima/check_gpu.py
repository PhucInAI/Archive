"""Check whether GPU work or not with both pytorch and tensorflow in current environment"""


# import torch
# import torch.nn as nn
# import torch.optim as optim
import tensorflow as tf
from tensorflow import keras # pylint: disable=no-name-in-module

# ########################################################################
# Pytorch check
# ########################################################################


# class SimpleModelPytorch(nn.Module):
#     """Simple Model Pytorch"""
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)


#     def forward(self, x):
#         """Feed forward function"""
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)

#         return x


# def test_pytorch(): # pylint: disable=too-many-locals
#     """Test Pytorch pipeline"""
#     print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
#     print("Check Pytorch GPU")
#     print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

#     # --------------------------------------------------------------------
#     # Define the training data
#     # --------------------------------------------------------------------
#     # pylint: disable=invalid-name
#     X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32) # pylint: disable = no-member
#     y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32) # pylint: disable = no-member


#     # --------------------------------------------------------------------
#     # Model setup
#     # --------------------------------------------------------------------
#     input_size = 2
#     hidden_size = 3
#     output_size = 1

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SimpleModelPytorch(input_size, hidden_size, output_size).to(device)

#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.1)

#     # --------------------------------------------------------------------
#     # Train the model
#     # --------------------------------------------------------------------
#     num_epochs = 1000
#     for epoch in range(num_epochs):
#         # ----------------------------------------------------------------
#         # Forward pass
#         # ----------------------------------------------------------------
#         inputs = X_train.to(device)
#         targets = y_train.to(device)
#         outputs = model(inputs) # pylint: disable=not-callable
#         loss = criterion(outputs, targets)

#         # ----------------------------------------------------------------
#         # Backward pass and optimization
#         # ----------------------------------------------------------------
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (epoch+1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#     # --------------------------------------------------------------------
#     # Test the trained model
#     # --------------------------------------------------------------------
#     with torch.no_grad():
#         test_input = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device) # pylint: disable = no-member
#         predicted = model(test_input) # pylint: disable=not-callable
#         print(predicted)


# ########################################################################
# Tensorflow check
# ########################################################################


class SimpleModelTensorflow(keras.Model): # pylint: disable=too-few-public-methods
    """Simple Modle Tensorflow"""
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(3, activation='relu')
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        """Feed forward function"""
        x = self.dense1(inputs)

        return self.dense2(x)


def test_tensorflow():
    """Test Tensorflow pipeline"""
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("Check Tensorflow GPU")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    # --------------------------------------------------------------------
    # Check if GPU is available
    # --------------------------------------------------------------------
    if tf.test.is_gpu_available():
        print("GPU is available")
    else:
        print("GPU is NOT available")
        return

    # --------------------------------------------------------------------
    # Define the training data
    # --------------------------------------------------------------------
    # pylint: disable=invalid-name
    X_train = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y_train = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

    # --------------------------------------------------------------------
    # Model setup
    # --------------------------------------------------------------------
    model = SimpleModelTensorflow()

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=0.1)

    # --------------------------------------------------------------------
    # Train the model
    # --------------------------------------------------------------------
    num_epochs = 1000
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            predictions = model(X_train)
            loss = loss_fn(y_train, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}')

    # --------------------------------------------------------------------
    # Test the trained model
    # --------------------------------------------------------------------
    predicted = model(X_train)
    print(predicted.numpy())



# ########################################################################
# Main
# ########################################################################


def main():
    """Main function"""
    # test_pytorch()
    test_tensorflow()


if __name__=="__main__":
    main()
