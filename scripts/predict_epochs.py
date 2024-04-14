# Try to predict the number of epochs to hit convergence
from sklearn.decomposition import PCA

from learning.bunch_learn import bunch_learn
from scripts.data import create_dataframe

model_count = 20

results = bunch_learn(
    model_count=model_count,
    rule_number=30,
    learning_epochs=1000,
    verbose=False,
)

df = create_dataframe(results)

# Initial weights for layer_1 PCA 3 components (X,Y,Z)
initial_weights = [weights[0] for weights in results["snapshots"]]

pca = PCA(n_components=3)
pca.fit(initial_weights)

# Transform the initial weights
initial_weights = pca.transform(initial_weights)

df["initial_pca_x"] = initial_weights[:, 0]
df["initial_pca_y"] = initial_weights[:, 1]
df["initial_pca_z"] = initial_weights[:, 2]


# Remove rows with 1001 epochs
df = df[df["epoch_count"] != 1001]
