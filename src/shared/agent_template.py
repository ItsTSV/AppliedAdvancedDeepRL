import torch
from pathlib import Path
from src.shared.environment_manager import EnvironmentManager
from src.shared.wandb_wrapper import WandbWrapper


class TemplateAgent:
    """Agent that serves as a template for all other agents. It holds attributes that are common to all
    agents (environment, wandb...) and operations that are shared among agents (IO)."""

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper):
        """Initializes the template agent with environment, wandb wrapper and IO parameters.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Wandb wrapper for tracking and hyperparameter management.
        """
        # Environment, wandb
        self.env = environment
        self.wdb = wandb

        # File handling
        current_path = Path(__file__).resolve()
        self.project_root = current_path.parent.parent.parent

    def save_model(self, actor):
        """INFERENCE ONLY -- Saves state dict of the actor model"""
        dir_parameter = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")

        # Get save directory path
        abs_save_dir = self.project_root / dir_parameter

        # Create directory
        abs_save_dir.mkdir(parents=True, exist_ok=True)

        # Get paths
        model_path = (abs_save_dir / name).with_suffix(".pth")
        rms_name = name + "_rms"
        rms_path = (abs_save_dir / rms_name).with_suffix(".npz")

        # Save
        torch.save(actor.state_dict(), str(model_path))
        self.env.save_normalization_parameters(str(rms_path))

    def save_artifact(self):
        """INFERENCE ONLY -- Saves state dict to wandb"""
        dir_parameter = self.wdb.get_hyperparameter("save_dir")
        name = self.wdb.get_hyperparameter("save_name")

        # Get save directory path
        abs_save_dir = self.project_root / dir_parameter

        # Create directory
        abs_save_dir.mkdir(parents=True, exist_ok=True)

        # Get paths
        model_path = (abs_save_dir / name).with_suffix(".pth")
        rms_name = name + "_rms"
        rms_path = (abs_save_dir / rms_name).with_suffix(".npz")

        # Save to WDB (online)
        self.wdb.log_model(name, str(model_path))
        self.wdb.log_model(name + "_rms", str(rms_path))

    def load_model(self, actor, path):
        """INFERENCE ONLY -- Loads state dict of the actor model"""
        # Model
        model_path = self.project_root / path

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Load and eval
        actor.load_state_dict(torch.load(model_path, weights_only=True))
        actor.eval()

        # Rms
        rms_path = model_path.parent / (model_path.stem + "_rms.npz")

        if not rms_path.exists():
            raise FileNotFoundError(f"Normalization file not found: {rms_path}")

        self.env.load_normalization_parameters(str(rms_path))
