from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_distribution_plot(df: pd.DataFrame, name: str, output_dir: Path):
    """Generates a plot that maps distribution of agent rewards.

    Args:
        df (pd.DataFrame): DataFrame with trial-reward-steps data
        name (str): Name of the tested model
        output_dir (Path): Absolute path to save directory
    """
    # General seaborn settings
    sns.set_theme(style="darkgrid", palette="muted")
    palette = sns.color_palette()

    # Data processing
    mean = df["Reward"].mean()
    median = df["Reward"].median()

    # Create plot
    plot = sns.displot(
        data=df,
        x="Reward",
        fill=True,
        height=8,
        aspect=1.5,
        color=palette[0],
        bins=8 if len(df) > 50 else 4
    )
    plot.figure.suptitle(f"Reward Distribution of {name} (N = {len(df)})")
    plot.set_axis_labels("Reward", "Number of Trials")

    # Add lines
    plt.axvline(mean, color=palette[1], linestyle="solid", label=f"Mean {mean:.2f}")
    plt.axvline(median, color=palette[3], linestyle="solid", label=f"Median {median:.2f}")
    plt.legend()

    # Save it to outputs
    plot.tight_layout()
    save_path = output_dir / f"{name}_distribution.png"
    plot.figure.savefig(save_path)
    plt.close()


def generate_scatter_plot(df: pd.DataFrame, name: str, output_dir: Path):
    """Generates a plot that maps steps taken - reward received relation.

    Args:
        df (pd.DataFrame): DataFrame with trial-reward-steps data
        name (str): Name of the tested model
        output_dir (Path): Absolute path to save directory
    """
    # General seaborn settings
    sns.set_theme(style="darkgrid", palette="muted")

    # Create plot
    plot = sns.scatterplot(
        data=df,
        x="Steps",
        y="Reward",
    )
    plot.set_title(f"Step-Reward Distribution of {name} (N = {len(df)})")

    # Save it to outputs
    save_path = output_dir / f"{name}_scatter.png"
    plot.figure.savefig(save_path)
    plt.close()
