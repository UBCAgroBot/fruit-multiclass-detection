import random

from common.autograd import Value
from common.MLP import MLP


# Helper to convert autograd Value nodes to plain Python floats for printing.
def to_scalar(v: Value) -> float:
    return float(v.data)


def main() -> None:
    # Phase 1: Define the supervised training data.
    # XOR truth table: [x1, x2] -> y
    dataset: list[tuple[list[float], float]] = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    # Phase 2: Initialize model + training hyperparameters.
    random.seed(42)
    model = MLP(nin=2, nouts=[3, 1])  # 2 inputs -> hidden layer(3) -> output(1)
    # Start from smaller weights for more stable early training.
    for p in model.parameters():
        p.data *= 0.1
    lr = 0.01  # learning rate for optimizer updates
    epochs = 3000  # full passes through the dataset

    # Phase 3: Train with full-batch gradient descent.
    # Optimizer: manual SGD-style parameter update (p.data -= lr * p.grad).
    # Loss: sum of squared errors over all samples (MSE-style objective without averaging).

    for epoch in range(epochs):
        # Reset epoch loss accumulator.
        loss = Value(0.0)
        for x_raw, y_raw in dataset:
            # Forward phase: wrap raw inputs/target into Value nodes for autograd.
            x = [Value(x_raw[0]), Value(x_raw[1])]
            y = Value(y_raw)
            pred = model(x)
            # Loss phase: accumulate squared error for this sample.
            loss = loss + (pred - y) ** 2

        # Backward phase: clear old gradients, then compute fresh gradients.
        for p in model.parameters():
            p.grad = 0
        loss.backward()
        # Optimization phase: gradient clipping + SGD update step.
        for p in model.parameters():
            p.grad = p.grad.clip(-1.0, 1.0)
            p.data -= lr * p.grad

        # Monitoring phase: print training loss periodically.
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:4d} loss={to_scalar(loss):.6f}")

    # Phase 4: Evaluate trained model on the XOR dataset.
    print("\nXOR predictions:")
    for x_raw, y_raw in dataset:
        x = [Value(x_raw[0]), Value(x_raw[1])]
        pred = model(x)
        pred_value = to_scalar(pred)
        # Convert regression output to binary class using 0.5 threshold.
        pred_label = 1 if pred_value > 0.5 else 0
        print(f"x={x_raw} target={int(y_raw)} pred={pred_value:.4f} class={pred_label}")


if __name__ == "__main__":
    main()

    # ACHIVES 100 percent accuracry yay
    # NOTE the gradient clipping was a must though because the grads were blowing up.
