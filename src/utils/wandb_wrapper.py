from omegaconf import OmegaConf
import wandb


class WandbWrapper:
    """Provides a wrapper that handles wandb configuration and logging.

    Attributes:
        wandb_parameters (dict): parameters that control the wandb configuration
        hyperparameters (dict): hyperparameters that control agent learning and behaviour
        run (wandb.Run): object that handles computations and logging
    """

    def __init__(self, yaml_path: str, mode="online"):
        """Initializes the instance with given configuration

        Args:
            yaml_path (str): location of the configuration file
            mode (str, optional): "online" for dashboard logging, "offline" for local only (no need to have account),
                                  "disabled" for no logging
        """
        yaml = OmegaConf.load(yaml_path)
        self.wandb_parameters = OmegaConf.to_container(yaml, resolve=True)
        self.__validate_parameters()

        self.hyperparameters = self.wandb_parameters.pop("config")
        self.run = wandb.init(
            **self.wandb_parameters,
            config=self.hyperparameters,
            settings=wandb.Settings(quiet=True),
            mode=mode
        )
        self.run.define_metric("*", step_metric="Total Steps")

    def get_hyperparameter(self, name: str):
        """Gets hyperparameter from given configuration"""
        if name in self.hyperparameters:
            return self.hyperparameters[name]
        raise KeyError("Hyperparameter not set!")

    def log(self, data: dict) -> None:
        self.run.log(data)

    def log_model(self, name: str, path: str) -> None:
        """Saves model to wandb"""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.run.finish()

    def __validate_parameters(self) -> None:
        """Validates whether the configuration file contains project, name and config specifications

        Raises:
            ValueError: When the configuration file is not valid
        """
        needed_fields = ["config", "project", "name"]
        if any(key not in self.wandb_parameters.keys() for key in needed_fields):
            raise ValueError(
                "The configuration file is invalid! Make sure it contains config, project and name!"
            )
