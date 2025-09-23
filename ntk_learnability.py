import warnings

import numpy as np
import torch
from nn_lib.analysis.ntk import ntk_task
from torch import nn


def pairs_of_batches(dataloader, include_ii=False):
    """Yield off-diagonal pairs of batches from a dataloader.

    Usage:

        for (i, x_i, y_i), (j, x_j, y_j) in pairs_of_batches(dataloader):

    If include_ii is True, this will include i==j pairs. If not, only lower-triangular (j<i) pairs
    will be included.
    """
    for i, batch_i in enumerate(dataloader):
        for j, batch_j in enumerate(dataloader):
            if j > i:
                break
            if include_ii or j < i:
                yield (i, *batch_i), (j, *batch_j)


def estimate_model_task_alignment(
    model: nn.Module,
    loss_fn: nn.Module,
    data: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
):
    """Estimate the learnability (rate of loss improvement) of a model on a dataset using NTK."""
    model.train()
    alignment_moment1, alignment_moment2 = [], []
    total_pairs = 0
    for (i, x_i, y_i), (j, x_j, y_j) in pairs_of_batches(data, include_ii=True):
        mask = torch.ones(len(x_i), len(x_j), device=device)
        if i == j:
            mask = torch.tril(mask, diagonal=-1)
        x_i, y_i, x_j, y_j = (
            x_i.to(device),
            y_i.to(device),
            x_j.to(device),
            y_j.to(device),
        )
        with warnings.catch_warnings():
            inner_l_times_k = ntk_task(model, loss_fn, x_i, y_i, x_j, y_j).detach() * mask
        alignment_moment1.append(inner_l_times_k.sum().item())
        alignment_moment2.append((inner_l_times_k**2).sum().item())
        total_pairs += mask.sum().item()

    avg_alignment = np.sum(alignment_moment1) / total_pairs
    var_alignment = np.sum(alignment_moment2) / total_pairs - avg_alignment**2
    mcse_alignment = np.sqrt(var_alignment / total_pairs)
    return avg_alignment, mcse_alignment


def estimate_local_learnability(
    model: nn.Module,
    loss_fn: nn.Module,
    data: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
):
    """Estimate the local learnability of a model on a dataset using NTK. This means estimating
    the current loss (± error) as well as the expected rate of loss improvement (± error) on the
    dataset.
    """
    # Do a pass over the data to get current loss
    with torch.no_grad():
        tmp, loss_fn.reduction = loss_fn.reduction, "none"
        initial_losses = []
        for x, y in data:
            x, y = x.to(device), y.to(device)
            output = model(x)
            initial_losses.extend(loss_fn(output, y).cpu().numpy())
        loss_fn.reduction = tmp
        loss = np.mean(initial_losses)
        loss_mcse = np.std(initial_losses) / np.sqrt(len(initial_losses))

    # Do another pass over the data to get model-task alignment (which, multiplied by a learning
    # rate, gives the expected rate of loss improvement)
    rate_of_change_of_loss, rate_of_change_of_loss_mcse = estimate_model_task_alignment(
        model, loss_fn, data, device
    )
    return loss, loss_mcse, rate_of_change_of_loss, rate_of_change_of_loss_mcse


# def postprocess_run(run: pd.Series, model: nn.Module = None):
#     if model is None:
#         model = new_model(
#
#         # Restore model from the parent run and data from the original version of this child run.
#         model.load_state_dict(load_artifact("weights.pt", run_id=run["tags.mlflow.parentRunId"]))
#         data = gen_data(
#             "a*x+b",
#             {"a": float(run["params.a"]), "b": float(run["params.b"])},
#             input_range=tuple(
#                 map(float, run["params.input_range"].strip("()").split(","))
#             ),
#             n_data=int(run["params.n_data"]),
#             seed=int(run["params.seed"]),
#         )
#
#         loss, loss_mcse, learning_speed, learning_speed_mcse = (
#             estimate_local_learnability(
#                 model,
#                 nn.MSELoss(),
#                 torch.utils.data.DataLoader(data, batch_size=10),
#                 device=torch.device("cpu"),
#             )
#         )
#
#         # Calling start_run with an existing run_id will resume that run and allow us to add new
#         # metrics/artifacts to it
#         with mlflow.start_run(run_id=run.run_id):
#             mlflow.log_metrics(
#                 {
#                     "ntk_learnability": learning_speed,
#                     "ntk_learnability_mcse": learning_speed_mcse,
#                     "initial_loss": loss,
#                     "initial_loss_mcse": loss_mcse,
#                 },
#                 step=0,
#             )
#
#
# if __name__ == "__main__":
#     import jsonargparse
#
#     parser = jsonargparse.ArgumentParser()
#     parser.add_function_arguments(run_main)
#     parser.add_argument("--config", action="config")
#     args = parser.parse_args()
#
#     # MLFlow setup
#     mlflow.set_tracking_uri("/data/projects/learnability/mlruns")
#     mlflow.set_experiment("2x+2")
#
#     # Find all runs from the 'run.py' script using the given parameters
#     params = args.as_dict()
#     params.pop("config")
#     if args.seed is None:
#         params.pop("seed")
#
#     prior_runs = search_runs_by_params(
#         experiment_name="2x+2", params=params, finished_only=True
#     )
#     model = new_model()
#     for _, run in tqdm(prior_runs.iterrows(), desc="Postprocessing"):
#         preexisting_value = run.get("metrics.ntk_learnability")
#         if (
#             preexisting_value is not None
#             and not np.isnan(preexisting_value)
#             and not np.isinf(preexisting_value)
#         ):
#             continue
#         postprocess_run(run, model)
#     print("Done.")
