"""
Question: 
    How do the initial weights impact the learning process?

Experiment: 
    Train ten models on the same rule number with \different initial weights.
    Then compare the number of epochs, starting loss, and final loss.
    
Conclusion:
    Initial weights greatly impact the learning process. Models with different 
    starting weights may converge to different solutions, and sometimes a model
    fails to converge at all.
"""

import pandas as pd

from learning.bunch_learn import bunch_learn

learning, total_snapshots, total_loss_records, total_gradient_norms = bunch_learn(
    model_count=10, rule_number=30, learning_epochs=1000, verbose=False
)


def main():
    epoch_counts = [len(losses) for losses in total_loss_records]
    starting_loss = [losses[0] for losses in total_loss_records]
    final_loss = [losses[-1] for losses in total_loss_records]
    model_number = [i for i in range(10)]

    df = pd.DataFrame(
        {
            "Model": model_number,
            "Epochs": epoch_counts,
            "Starting Loss": starting_loss,
            "Final Loss": final_loss,
        }
    )

    print(df)


if __name__ == "__main__":
    main()
