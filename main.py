# importing dependencies
import numpy as np

# loading mnist dataset
data = np.load('mnist.npz', allow_pickle=True)

# which arrays we have in dataset
print(data.files)

# splitting oyr arrays
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# printing the shape of dataset
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# converting 28*28 to a vector
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

# normalizing data
x_train = x_train/255
x_test = x_test/255

# checking the number of classes
num_classes = len(np.unique(y_train))
print("number of classes", num_classes)

# one hot encoding labels
y_train = np.zeros((y_train.size, num_classes))
y_train[np.arange(y_test.size), y_test] = 1
print(y_train.shape)
print(y_test.shape)

m=x_train.shape[0]
input_layer = x_train.shape[1]
hidden_layer = 64
output_layer = 1 if num_classes == 2 else num_classes
epochs = 100
learning_rate = 0.0001

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivated(z):
    a = sigmoid(z)
    return a * (1 - a)

w1 = np.random.randn(hidden_layer, input_layer)
print(f"w1: {w1}")
print(f"shape w1: {w1.shape}")

b1 = np.zeros((hidden_layer, 1))
print(f"b1: {b1}")
print(f"b1: {b1.shape}")


w2 = np.random.randn(output_layer, hidden_layer)
print(f"w2: {w2}")
print(f"shape w2: {w2.shape}")

b2 = np.zeros((output_layer, 1))
print(f"b2: {b2}")
print(f"shape b2: {b2.shape}")


for epoch in range(1, epochs+1):

    #forward
    Z1 = np.dot(w1, x_train.T) + b1
    print("Z1 shape:", Z1.shape)
    A1 = sigmoid(Z1)
    print("A1 shape:", A1.shape)
    Z2 = np.dot(w2, A1) + b2
    print("Z2 shape:", Z2.shape)
    A2 = sigmoid(Z2)
    print("A2 shape:", A2.shape)

    #loss
    if epoch%10==0:
        loss = np.mean((A2-y_train.T)**2)

    #backward
    dZ2 = A2 - y_train.T
    print("dZ2 shape:", dZ2.shape)

    dw2 = np.dot(dZ2, A1.T) / m
    print("dW2 shape:", dw2.shape)  # (1,2)

    db2 = np.sum(dZ2, axis=1, keepdims=True)
    print("db2 shape:", db2.shape)  # (1,1)

    dZ1 = np.dot(w2.T, dZ2) * sigmoid_derivated(Z1)
    print("dZ1 shape:", dZ1.shape)  # (2,4)

    dw1 = np.dot(dZ1, x_train) / m
    print("dW1 shape:", dw1.shape)  # (2,2)

    db1 = np.sum(dZ1, axis=1, keepdims=True)
    print("db1 shape:", db1.shape)  # (2,1)

    #update
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2


x_new = x_test[0].reshape(784,1)
Z1 = np.dot(w1, x_new) + b1
print("Z1 shape:", Z1.shape)
A1 = sigmoid(Z1)
print("A1 shape:", A1.shape)
Z2 = np.dot(w2, A1) + b2
print("Z2 shape:", Z2.shape)
A2 = sigmoid(Z2)
print("A2 shape:", A2.shape)
predicted_label = np.argmax(A2)
print("predicted labe:", predicted_label)

