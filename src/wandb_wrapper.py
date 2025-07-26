from omegaconf import OmegaConf
import wandb


class WandbWrapper:
    """Provides a wrapper that handles wandb configuration and logging.

    Attributes:
        wandb_parameters (dict): parameters that control the wandb configuration
        hyperparameters (dict): hyperparameters that control agent learning and behaviour
        run (wandb.Run): object that handles computations and logging
    """

    def __init__(self, yaml_path: str) -> None:
        """Initializes the instance with given configuration

        Args:
            yaml_path (str): location of the configuration file
        """
        yaml = OmegaConf.load(yaml_path)
        self.wandb_parameters = OmegaConf.to_container(yaml, resolve=True)
        self.__validate_parameters()

        self.hyperparameters = self.wandb_parameters.pop("config")
        self.run = wandb.init(
            **self.wandb_parameters,
            config=self.hyperparameters,
            settings=wandb.Settings(quiet=True)
        )

    def log(self, data: dict) -> None:
        self.run.log(data)

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
