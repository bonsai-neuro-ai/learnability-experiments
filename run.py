from typing import Optional

import mlflow
import numpy as np
import torch
from nn_lib.utils import save_as_artifact, load_artifact
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import trange

from ntk_learnability import estimate_local_learnability


def new_model(seed: int = -1) -> nn.Module:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def gen_data(
    formula: str = "a*x+b",
    params: Optional[dict] = None,
    input_range: tuple[float, float] = (-50, 50),
    n_data: int = 1000,
    seed: int = -1,
):
    rng = torch.Generator().manual_seed(seed)
    x = (
        torch.rand(n_data, 1, generator=rng) * (input_range[1] - input_range[0])
        + input_range[0]
    )

    try:
        y = eval(formula, {"x": x, **(params or {})})
    except NameError as e:
        raise ValueError("Formula contains variables not provided as params") from e

    return TensorDataset(x, y)


def train(model, data_train, epochs, batch_size, lr):
    model.train()

    init_loss, init_loss_mcse, learning_speed, learning_speed_mcse = (
        estimate_local_learnability(
            model, nn.MSELoss(), DataLoader(data_train, batch_size=len(data_train))
        )
    )
    mlflow.log_metrics(
        {
            "init_loss": init_loss,
            "init_loss_mcse": init_loss_mcse,
            "ntk_learning_speed": learning_speed,
            "ntk_learning_speed_mcse": learning_speed_mcse,
        }
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True
    )
    step = 0
    for _ in trange(epochs, leave=False, desc="Training epochs"):
        for x, y in loader:
            optim.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()

            mlflow.log_metric("loss", loss.item(), step=step)
            step += 1


def main(
    input_range: tuple[float, float] = (-50, 50),
    n_data: int = 1000,
    initial_epochs: int = 100,
    initial_batch_size: int = 32,
    finetune_epochs: int = 75,
    finetune_batch_size: int = 5,
    lr: float = 1e-3,
    seed: Optional[int] = None,
):
    data = gen_data(
        formula="a*x+b",
        params={"a": 2, "b": 2},
        input_range=input_range,
        n_data=n_data,
        seed=seed,
    )

    model = new_model(seed)
    mlflow.log_params({"phase": "initial_training", "a": 2.0, "b": 2.0})
    train(model, data, initial_epochs, initial_batch_size, lr=lr)
    save_as_artifact(model.state_dict(), "weights.pt")

    def restore_model_to_initial_training_result():
        model.load_state_dict(load_artifact("weights.pt"))

    # Test generalization type 1a [2x+2 --> ax+2]
    # At time 0, expected error is E[(ax+2) - (2x+2)]^2 = E[((a-2)x)^2] = (a-2)^2(v^2)/3 for x~U(-v,+v)
    # For v=10 and a=3 this is 100/3 ~= 33.33
    restore_model_to_initial_training_result()
    with mlflow.start_run(nested=True):
        mlflow.log_params({"phase": "1a", "a": 3.0, "b": 2.0})
        data = gen_data(
            formula="a*x+b",
            params={"a": 3.0, "b": 2.0},
            input_range=input_range,
            n_data=n_data,
            seed=seed + 1000,
        )
        train(model, data, finetune_epochs, finetune_batch_size, lr=lr)

    # Test generalization type 1b [2x+2 --> 2x+b]
    # At time 0, expected error is E[(2x+b) - (2x+2)]^2 = E[(b-2)^2] = (b-2)^2
    # To make this match 33.33, set b=sqrt(33.33)+2 ~= 7.77
    b = np.sqrt(100 / 3) + 2
    restore_model_to_initial_training_result()
    with mlflow.start_run(nested=True):
        mlflow.log_params({"phase": "1b", "a": 2.0, "b": b})
        data = gen_data(
            formula="a*x+b",
            params={"a": 2.0, "b": b},
            input_range=input_range,
            n_data=n_data,
            seed=seed + 2000,
        )
        train(model, data, finetune_epochs, finetune_batch_size, lr=lr)

    # Test generalization type 2a [2(x+1) --> c(x+1)]
    # At time 0, expected error is E[(a(x+1) - 2(x+1))^2] = E[(c-2)(x+1)^2] = (c-2)((v^2)/3 + 1) for x~U(-v,+v)
    # To make this match 33.33, set a=b=c = (100/103) + 2 ~= 2.97
    c = (100 / 103) + 2
    restore_model_to_initial_training_result()
    with mlflow.start_run(nested=True):
        mlflow.log_params({"phase": "2a", "a": c, "b": c})
        data = gen_data(
            formula="a*x+b",
            params={"a": c, "b": c},
            input_range=input_range,
            n_data=n_data,
            seed=seed + 3000,
        )
        train(model, data, finetune_epochs, finetune_batch_size, lr=lr)

    # Test generalization type 2b [2(x+1) --> 2(x+d)]
    # At time 0, expected error is E[(2(x+d) - 2(x+1))^2] = E[(2(d-1))^2] = 4(d-1)^2
    # To make this match 33.33, set d = sqrt(100 / 12) + 1 = 3.89
    d = np.sqrt(100 / 12) + 1
    restore_model_to_initial_training_result()
    with mlflow.start_run(nested=True):
        mlflow.log_params({"phase": "2b", "a": 2.0, "b": 2 * d})
        data = gen_data(
            formula="a*x+b",
            params={"a": 2.0, "b": 2 * d},
            input_range=input_range,
            n_data=n_data,
            seed=seed + 4000,
        )
        train(model, data, finetune_epochs, finetune_batch_size, lr=lr)


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")
    args = parser.parse_args()

    # Post-processing and default params
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)

    # MLFlow setup
    mlflow.set_tracking_uri("/data/projects/learnability/mlruns")
    mlflow.set_experiment("2x+2")

    params = args.as_dict()
    params.pop("config")
    with mlflow.start_run():
        mlflow.log_params(params)
        main(**params)
