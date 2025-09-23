import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import torch
from nn_lib.utils import load_artifact
from torch import nn

from run import new_model

mlflow.set_tracking_uri("/data/projects/learnability/mlruns")
df = mlflow.search_runs(experiment_names=["2x+2"])
initial_runs = df[df["params.phase"] == "initial_training"]
finetuning_runs = df[df["params.phase"].isin(["1a", "1b", "2a", "2b"])]

# %%


def load_metric_history(run_id, metric, client=None):
    if client is None:
        client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, metric)
    return pd.DataFrame([{"step": m.step, metric: m.value} for m in history]).set_index(
        "step"
    )


def model_from_run(run_id) -> nn.Module:
    model = new_model()
    model.load_state_dict(load_artifact(run_id=run_id, path="weights.pt"))
    return model


# %% How well did they all initially learn 2x+2?

x = torch.linspace(-50, 50, 1000).unsqueeze(1)
true_y = 2 * x + 2

plt.figure()
plt.plot(
    x.squeeze(),
    true_y.squeeze(),
    label="true",
    color="black",
    linewidth=2,
    linestyle="--",
)
for _, run in initial_runs.iterrows():
    model = model_from_run(run.run_id)
    with torch.no_grad():
        pred_y = model(x).squeeze()
    plt.plot(x.squeeze(), pred_y, label=run.run_id)
plt.ylim(-100, 104)
plt.title("Learned 2x+2 approximations")
plt.show()

plt.figure()
for _, run in initial_runs.iterrows():
    history = load_metric_history(run.run_id, "loss")
    plt.plot(history.index, history["loss"], linewidth=0.5)
plt.yscale("log")
plt.title("Initial training loss")
plt.show()

plt.figure()
for _, run in finetuning_runs.iterrows():
    history = load_metric_history(run.run_id, "loss")
    plt.plot(history.index, history["loss"], linewidth=0.5)
plt.yscale("log")
plt.title("Finetuning training loss")
plt.show()

# %% Is there any systematic correlation between (1a, 1b) and (2a, 2b) performance?

# Start by pivoting the data so that the separate rows for 1a/1b/2a/2b become columns, grouped
# by their parent run (the initial training run)
per_phase_losses = finetuning_runs.pivot(
    index="tags.mlflow.parentRunId", columns="params.phase", values="metrics.loss"
)

fig = plt.figure(figsize=(12, 12))
sns.pairplot(data=per_phase_losses)
plt.show()

# %%

plt.figure()
sns.scatterplot(
    finetuning_runs, x="metrics.init_loss", y="metrics.loss", hue="params.phase"
)
plt.xscale("log")
plt.yscale("log")
plt.title("Final vs initial finetuning loss")
plt.show()

# %%

finetuning_runs["projected_loss"] = (
    finetuning_runs["metrics.init_loss"]
    - finetuning_runs["metrics.ntk_learning_speed"] * 0.001
)
plt.figure()
sns.scatterplot(
    finetuning_runs, x="metrics.init_loss", y="projected_loss", hue="params.phase"
)
plt.xscale("log")
plt.yscale("log")
plt.title("Projected vs initial finetuning loss")
plt.show()

plt.figure()
sns.scatterplot(
    finetuning_runs, x="metrics.loss", y="projected_loss", hue="params.phase"
)
plt.xscale("log")
plt.yscale("log")
plt.title("Actual vs projected finetuning loss")
plt.show()
