#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import struct
from array import array
from os.path import join
from typing import Any, List, Tuple, Union

import numpy as np  # linear algebra

from common.autograd import Value


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath: str,
        training_labels_filepath: str,
        test_images_filepath: str,
        test_labels_filepath: str,
    ) -> None:
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(
        self, images_filepath: str, labels_filepath: str
    ) -> Tuple[List[Any], Any]:
        labels: Union[List[int], "array[int]"] = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            image_data = array("B", file.read())

        images: List[Any] = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self) -> Tuple[Tuple[List[Any], Any], Tuple[List[Any], Any]]:
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


# Verify Reading Dataset via MnistDataloader class
#


#
# Set file paths based on added MNIST Datasets
#
input_path = "src/common/DataMNIST/raw"
training_images_filepath = join(input_path, "train-images-idx3-ubyte")
training_labels_filepath = join(input_path, "train-labels-idx1-ubyte")
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte")


#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

### MNIST CODE


class Linear:
    def __init__(self, nin: int, nout: int):
        std = np.sqrt(2.0 / nin)
        self.W = Value(np.random.randn(nin, nout) * std)
        self.b = Value(np.zeros(nout))

    def __call__(self, x: Value) -> Value:
        return (x @ self.W) + self.b

    def parameters(self) -> List[Value]:
        return [self.W, self.b]


class MnistNetwork:
    def __init__(self) -> None:
        self.l1 = Linear(784, 100)
        self.l2 = Linear(100, 100)
        self.l3 = Linear(100, 50)
        self.l4 = Linear(50, 10)

    def __call__(self, x: Value) -> Value:
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x).relu()
        return self.l4(x)

    def parameters(self) -> List[Value]:
        return (
            self.l1.parameters()
            + self.l2.parameters()
            + self.l3.parameters()
            + self.l4.parameters()
        )


def cross_entropy_loss(logits: Value, targets: np.ndarray) -> Value:
    max_logits = logits.max(axis=1, keepdims=True)
    logits_safe = logits - max_logits
    counts = logits_safe.exp()
    counts_sum = counts.sum(axis=1, keepdims=True)
    probs = counts / counts_sum
    log_probs = probs.log()
    targets_val = Value(targets)
    loss: Value = -(targets_val * log_probs).sum() / logits.data.shape[0]
    return loss


if __name__ == "__main__":
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 1. Load the data using Rowan's class
    print("Loading Data...")
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # 2. Format the training data
    print("Formatting Data...")
    # Flatten 28x28 into 784 and normalize to 0.0 - 1.0
    X_train = np.array(x_train).reshape(-1, 784) / 255.0

    # One-hot encode the labels using NumPy
    num_classes = 10
    y_train_arr = np.array(y_train)
    Y_train_one_hot = np.eye(num_classes)[y_train_arr]

    # --- VISUALIZATION 1: Raw Data ---
    print("Saving Raw Data Visualization...")
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        img = X_train[i].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label: {y_train_arr[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("mnist_raw_data_samples.png")
    plt.close()  # Free up memory

    # 3. Initialize Model & Hyperparameters
    model = MnistNetwork()
    epochs = 10
    batch_size = 64
    lr = 0.1
    num_samples = X_train.shape[0]

    print("Starting Training...")

    # --- SETUP FOR VISUALIZATION 3: Training Curve ---
    history_loss = []

    # 4. The Training Loop
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train_one_hot[indices]

        epoch_loss = 0.0
        batches = 0

        for i in range(0, num_samples, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i : i + batch_size]
            Y_batch = Y_shuffled[i : i + batch_size]

            # Forward Pass
            x_val = Value(X_batch)
            logits = model(x_val)
            loss = cross_entropy_loss(logits, Y_batch)

            # Zero Gradients
            for p in model.parameters():
                p.grad = np.zeros_like(p.data)

            # Backward Pass
            loss.backward()

            # SGD Update Step
            for p in model.parameters():
                p.grad = np.clip(p.grad, -1.0, 1.0)  # Gradient clipping
                p.data -= lr * p.grad

            epoch_loss += float(loss.data.item())
            batches += 1

        avg_loss = epoch_loss / batches
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

        # Track loss for plotting
        history_loss.append(avg_loss)

    # --- VISUALIZATION 3: Save Training Curve ---
    print("Saving Training Curve Visualization...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), history_loss, marker="o", linestyle="-", color="b")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True)
    plt.savefig("mnist_training_curve.png")
    plt.close()

    # 5. Evaluate Accuracy on the Test Set
    print("\nEvaluating Test Accuracy...")

    # Format the test data exactly like the training data
    X_test = np.array(x_test).reshape(-1, 784) / 255.0
    y_test_arr = np.array(y_test)

    correct_predictions = 0
    total_test_samples = X_test.shape[0]

    # --- SETUP FOR VISUALIZATION 4: Predictions ---
    all_predictions = []

    # Run through the test set in batches (keeps memory usage low)
    for i in range(0, total_test_samples, batch_size):
        # Get mini-batch
        X_batch = X_test[i : i + batch_size]
        y_batch = y_test_arr[i : i + batch_size]

        # Forward pass (no need to calculate gradients here!)
        x_val = Value(X_batch)
        logits = model(x_val)

        # The prediction is the index of the highest logit
        predictions = np.argmax(logits.data, axis=1)

        # Track predictions for the confusion matrix
        all_predictions.extend(predictions)

        # Count how many predictions match the true label
        correct_predictions += np.sum(predictions == y_batch)

    accuracy = (correct_predictions / total_test_samples) * 100
    print(f"Final Test Accuracy: {accuracy:.2f}%")

    # --- VISUALIZATION 4: Save Confusion Matrix ---
    print("Saving Confusion Matrix...")
    cm = confusion_matrix(y_test_arr, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("mnist_confusion_matrix.png")
    plt.close()

    # --- SAVE METRICS TO FILE ---
    print("Saving Metrics to JSON...")
    metrics = {"final_accuracy": accuracy, "loss_per_epoch": history_loss}

    with open("mnist_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nDone! Check your folder for the new .png and .json files.")
