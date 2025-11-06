import numpy as np
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    RadioSet,
    RadioButton,
    Markdown,
    Checkbox,
    Input,
    Button,
    RichLog
)
from textual.validation import Regex
import glob
from wandb_wrapper import WandbWrapper
from environment_manager import EnvironmentManager
from ppo_agent_continuous import PPOAgentContinuous
from ppo_models import ContinuousActorCriticNet
from sac_agent import SACAgent


class RlPlayground(App):
    """A Textual app to test RL agents"""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def __init__(self):
        super().__init__()
        self.config_path = None
        self.model_path = None
        self.render_mode = None
        self.num_trials = 1
        self.generate_csv_log = False
        self.generate_chart_report = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app"""
        # Header
        yield Header(show_clock=True)

        # Configuration file selector
        yield Markdown("# Select Configuration File")
        with RadioSet(id="config_selector"):
            for config_file in glob.glob("../config/*yaml"):
                yield RadioButton(config_file)

        # Model selector
        yield Markdown("# Select Model File")
        with RadioSet(id="model_selector"):
            for model_file in glob.glob("../models/*pth"):
                yield RadioButton(model_file)
            yield RadioButton("RANDOM POLICY")

        # Render settings
        yield Markdown("# Render Settings")
        with RadioSet(id="render_selector"):
            yield RadioButton("No Rendering")
            yield RadioButton("Human Rendering")
            yield RadioButton("Video Rendering")

        # Trial runner
        yield Markdown("# How many trials?")
        yield Input(
            placeholder="Enter number of trials",
            id="trial_input",
            validators=[
                Regex(regex=r"^\d+$", failure_description="Must be a positive integer")
            ],
        )

        # Additional settings
        yield Markdown("# Additional Settings")
        yield Checkbox("Generate CSV Log", id="csv_log_checkbox")
        yield Checkbox("Generate Chart Report", id="chart_report_checkbox")

        # Confirmation button
        yield Button("Confirm and run", id="run_button", variant="primary")

        # Debug text area
        yield RichLog(id="debug_output", markup=True, wrap=True)

        # Footer
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press event to run the trials"""
        if event.button.id == "run_button":
            # Get number of trials
            trial_input = self.query_one("#trial_input", Input)
            self.num_trials = int(trial_input.value) if trial_input.value and trial_input.is_valid else 1

            # Run trials
            self.run_worker(self.run_trials, exclusive=True, thread=True)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio set change events to log selections"""
        if event.radio_set.id == "config_selector":
            self.config_path = str(event.pressed.label)
        elif event.radio_set.id == "model_selector":
            self.model_path = str(event.pressed.label)
        elif event.radio_set.id == "render_selector":
            self.render_mode = str(event.pressed.label)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox change events to log additional settings"""
        if event.checkbox.id == "csv_log_checkbox":
            self.generate_csv_log = bool(event.value)
        elif event.checkbox.id == "chart_report_checkbox":
            self.generate_chart_report = bool(event.value)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode"""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def run_trials(self):
        """Run the specified number of trials with the selected configuration"""
        # Clear debug output
        debug_output = self.query_one(RichLog)
        debug_output.clear()

        # Validate selections
        if not all([self.config_path, self.model_path, self.render_mode]):
            debug_output.write("[bold red]Error:[/bold red] Please make all selections before running trials.")
            return

        # Write info
        debug_output = self.query_one(RichLog)
        debug_output.clear()
        debug_output.write(f"[bold green]Configuration File:[/bold green] {self.config_path}")
        debug_output.write(f"[bold green]Model File:[/bold green] {self.model_path}")
        debug_output.write(f"[bold green]Render Mode:[/bold green] {self.render_mode}")
        debug_output.write(f"[bold green]Number of Trials:[/bold green] {self.num_trials}")
        debug_output.write(f"[bold green]Generate CSV Log:[/bold green] {self.generate_csv_log}")
        debug_output.write(f"[bold green]Generate Chart Report:[/bold green] {self.generate_chart_report}")
        debug_output.write("[bold blue]Running trials...[/bold blue]")

        # Initialize WandbWrapper
        wdb = WandbWrapper(self.config_path, mode="disabled")

        # Initialize environment with correct settings
        name = wdb.get_hyperparameter("environment")
        env = EnvironmentManager(name, "human" if self.render_mode == "Human Rendering" else "rgb_array")
        env.build_continuous()
        if self.render_mode == "Video Rendering":
            env.build_video_recorder()

        # Initialize agent
        algorithm = wdb.get_hyperparameter("algorithm")
        if algorithm == "PPO Continuous":
            action_space, observation_space = env.get_dimensions()
            network_size = wdb.get_hyperparameter("network_size")
            model = ContinuousActorCriticNet(action_space, observation_space, network_size)
            agent = PPOAgentContinuous(env, wdb, model)
        elif algorithm == "SAC":
            agent = SACAgent(env, wdb)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Load model if not random policy
        if self.model_path != "RANDOM POLICY":
            try:
                agent.load_model(self.model_path)
            except:
                debug_output.write(f"[bold red]Make sure the model is compactible with selected agent![/bold red]")

        # Run trials
        rewards = []
        for i in range(self.num_trials):
            reward = agent.play()
            rewards.append(reward)
            debug_output = self.query_one(RichLog)
            debug_output.write(f"[bold yellow]Trial {i + 1} Reward:[/bold yellow] {reward}")

        average_reward = np.sum(rewards) / self.num_trials
        debug_output = self.query_one(RichLog)
        debug_output.write(f"[bold magenta]Average Reward:[/bold magenta] {average_reward}")

        # Will be added later: CSV and Chart export

        # Finish writeup
        debug_output.write(f"[bold blue]Trials finished[/bold blue]")

        # Clean
        env.close()
        wdb.finish()


if __name__ == "__main__":
    app = RlPlayground()
    app.run()
