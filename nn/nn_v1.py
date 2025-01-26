import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Dummy Data Creation
num_features = 20
num_classes = 3
num_samples = 1000
batch_size = 32
num_epochs = 200
learning_rate = 0.01

# Generate random data
X = mx.random.normal((num_samples, num_features))
y = mx.random.randint(0, num_classes, (num_samples,))

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNN(input_dim=num_features, hidden_dim=64, output_dim=num_classes)

# Loss function
def loss_fn(model, X, y):
    logits = model(X)
    return mx.mean(nn.losses.cross_entropy(logits, y))

# Define accuracy calculation
def accuracy_fn(model, X, y):
    preds = mx.argmax(model(X), axis=1)
    return mx.mean(preds == y)

# Gradient function
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Optimizer
optimizer = optim.SGD(learning_rate=learning_rate)

# Training loop
for epoch in range(num_epochs):
    perm = mx.array(np.random.permutation(X.shape[0]))
    X = X[perm]
    y = y[perm]

    for i in range(0, X.shape[0], batch_size):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]

        loss, grads = loss_and_grad_fn(model, batch_X, batch_y)
        optimizer.update(model, grads)

    # Evaluate on the training set
    acc = accuracy_fn(model, X, y)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

# Model evaluation after training
final_acc = accuracy_fn(model, X, y)
print(f"Final Training Accuracy: {final_acc.item():.4f}")