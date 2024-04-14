# Create dataframe from bunch training results

import pandas as pd
import torch

import weight_analysis.magnitude as magnitude


def create_dataframe(results: dict) -> pd.DataFrame:
    """
    Create a dataframe from the results of bunch training.

    Parameters
    ----------
    results : dict
        The results of bunch training.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the results of bunch training.
    """

    all_losses = []
    all_seeds = []
    all_snapshots = []
    all_gradients = []

    for model_id, model_results in results.items():
        all_losses.append(model_results["losses"])
        all_seeds.append(model_id)
        all_snapshots.append(model_results["snapshots"])
        all_gradients.append(model_results["gradients"])

    epoch_counts = [len(losses) for losses in all_losses]
    starting_loss = [losses[0] for losses in all_losses]
    final_loss = [losses[-1] for losses in all_losses]

    initial_weights = [weights[0] for weights in all_snapshots]
    final_weights = [weights[-1] for weights in all_snapshots]

    df = pd.DataFrame(
        {
            "seed": all_seeds,
            "epoch_count": epoch_counts,
            "starting_loss": starting_loss,
            "final_loss": final_loss,
        }
    )

    #! L1 initial weight stats
    df["l1_initial_magnitude"] = [
        magnitude.basic_magnitude(torch.tensor(weights)) for weights in initial_weights
    ]
    df["l1_initial_median"] = [
        torch.median(torch.tensor(weights)).item() for weights in initial_weights
    ]
    df["l1_initial_mean"] = [
        torch.mean(torch.tensor(weights)).item() for weights in initial_weights
    ]
    df["l1_initial_std"] = [
        torch.std(torch.tensor(weights)).item() for weights in initial_weights
    ]
    df["l1_initial_var"] = [
        torch.var(torch.tensor(weights)).item() for weights in initial_weights
    ]
    df["l1_initial_max"] = [
        torch.max(torch.tensor(weights)).item() for weights in initial_weights
    ]
    df["l1_initial_min"] = [
        torch.min(torch.tensor(weights)).item() for weights in initial_weights
    ]

    #! L1 final weight stats
    df["l1_final_magnitude"] = [
        magnitude.basic_magnitude(torch.tensor(weights)) for weights in final_weights
    ]
    df["l1_final_median"] = [
        torch.median(torch.tensor(weights)).item() for weights in final_weights
    ]
    df["l1_final_mean"] = [
        torch.mean(torch.tensor(weights)).item() for weights in final_weights
    ]
    df["l1_final_std"] = [
        torch.std(torch.tensor(weights)).item() for weights in final_weights
    ]
    df["l1_final_var"] = [
        torch.var(torch.tensor(weights)).item() for weights in final_weights
    ]
    df["l1_final_max"] = [
        torch.max(torch.tensor(weights)).item() for weights in final_weights
    ]
    df["l1_final_min"] = [
        torch.min(torch.tensor(weights)).item() for weights in final_weights
    ]

    df["l1_magnitude_change"] = df["l1_final_magnitude"] - df["l1_initial_magnitude"]
    df["l1_magnitude_change_epoch"] = df["l1_magnitude_change"] / df["epoch_count"]

    df["initial_spectral_norm"] = [
        magnitude.spectral_norm(
            torch.tensor(weights).unsqueeze(0)
            if torch.tensor(weights).dim() == 1
            else torch.tensor(weights)
        )
        for weights in initial_weights
    ]
    df["final_spectral_norm"] = [
        magnitude.spectral_norm(
            torch.tensor(weights).unsqueeze(0)
            if torch.tensor(weights).dim() == 1
            else torch.tensor(weights)
        )
        for weights in final_weights
    ]
    df["spectral_norm_change"] = df["final_spectral_norm"] - df["initial_spectral_norm"]

    # L1 gradient stats
    df["l1_mean_gradient"] = [
        torch.mean(torch.tensor(gradient[0])).item() for gradient in all_gradients
    ]
    df["l1_median_gradient"] = [
        torch.median(torch.tensor(gradient[0])).item() for gradient in all_gradients
    ]
    df["l1_std_gradient"] = [
        torch.std(torch.tensor(gradient[0])).item() for gradient in all_gradients
    ]
    df["l1_var_gradient"] = [
        torch.var(torch.tensor(gradient[0])).item() for gradient in all_gradients
    ]

    # L1 gradient stats of the first 200 epochs
    df["l1_mean_gradient_200"] = [
        torch.mean(torch.tensor(gradient[0][:200])).item() for gradient in all_gradients
    ]
    df["l1_median_gradient_200"] = [
        torch.median(torch.tensor(gradient[0][:200])).item()
        for gradient in all_gradients
    ]
    df["l1_std_gradient_200"] = [
        torch.std(torch.tensor(gradient[0][:200])).item() for gradient in all_gradients
    ]
    df["l1_var_gradient_200"] = [
        torch.var(torch.tensor(gradient[0][:200])).item() for gradient in all_gradients
    ]

    return df
