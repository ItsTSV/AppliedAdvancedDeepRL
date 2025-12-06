from src.utils.arg_handler import get_args
from src.shared.wandb_wrapper import WandbWrapper
from src.shared.environment_manager import EnvironmentManager
from src.ppo.agent_continuous import PPOAgentContinuous
from src.sac.agent import SACAgent


if __name__ == "__main__":
    # Get arguments
    args = get_args()

    # Initialize WandbWrapper
    wdb = WandbWrapper(
        args.config,
        args.log
    )

    # Initialize environment
    name = wdb.get_hyperparameter("environment")
    env = EnvironmentManager(name, "rgb_array")
    env.build_continuous()

    # Initialize agent
    algorithm = wdb.get_hyperparameter("algorithm")
    if algorithm == "PPO Continuous":
        agent = PPOAgentContinuous(env, wdb)
    elif algorithm == "SAC":
        agent = SACAgent(env, wdb)
    else:
        raise ValueError("Please, select a valid algorithm! [PPO Continuous, SAC, TD3]")

    # Sanity check -- print configurations
    print("-" * 20)
    print(f"Training {algorithm} agent on {name} environment.")
    print(f"Agent type is {type(agent)}.")
    print(f"Logging is set to {args.log}.")
    print(f"Hyperparameters: {wdb.hyperparameters}")
    print("-" * 20)

    # Start training
    agent.train()

    # Cleanup
    env.close()
    wdb.finish()
