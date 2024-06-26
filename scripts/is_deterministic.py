"""
Question: 
    Is the learning process deterministic?

Experiment: 
    Train two models on the same rule number with the same seed so that the initial 
    weights are the same. Then compare the final weights, loss records, and gradient norms.
    
Conclusion:
    The two learning processes are deterministic, given the same initial weights, 
    seed, and hyperparameters.
"""

import torch

from learning.bunch_learn import bunch_learn

# Set seed for reproducibility
seed = 1234
model_count = 20

results = bunch_learn(
    model_count=model_count,
    rule_number=30,
    learning_epochs=2000,
    seed=seed,
    verbose=False,
)

total_snapshots = results["snapshots"]
total_loss_records = results["losses"]
total_gradient_norms = results["gradients"]


def main():
    # Assert same initial weights
    assert torch.allclose(
        torch.tensor(total_snapshots[0][0]), torch.tensor(total_snapshots[1][0])
    )
    print("Initial weights are the same.")

    # Asert same final weights
    assert torch.allclose(
        torch.tensor(total_snapshots[0][-1]), torch.tensor(total_snapshots[1][-1])
    )
    print("Final weights are the same.")

    # Check if the loss records are the same
    assert total_loss_records[0] == total_loss_records[1]
    print("Loss records are the same.")

    # Check if the gradient norms are the same
    assert total_gradient_norms[0] == total_gradient_norms[1]
    print("Gradient norms are the same.")

    print("Two learning processes are deterministic.")


if __name__ == "__main__":
    main()
    from charts.gradient import gradient_w_loss

    first_layer_gradients = [gradient[0] for gradient in total_gradient_norms[10]]

    gradient_w_loss(
        first_layer_gradients,
        total_loss_records[10],
        title="Layer Gradients and Loss: N=5000",
    )
