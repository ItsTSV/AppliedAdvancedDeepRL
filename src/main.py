from src.utils.arg_handler import get_args
from src.shared.wandb_wrapper import WandbWrapper
from src.shared.environment_manager import EnvironmentManager
from src.ppo.agent_continuous import PPOAgentContinuous
from src.sac.agent import SACAgent
from src.td3.agent import TD3Agent


if __name__ == "__main__":
    args = get_args()

    wdb = WandbWrapper(args.config, args.log)

    name = wdb.get_hyperparameter("environment")
    env = EnvironmentManager(name, "rgb_array")
    env.build_continuous()

    algorithm = wdb.get_hyperparameter("algorithm")
    if algorithm == "PPO Continuous":
        agent = PPOAgentContinuous(env, wdb)
    elif algorithm == "SAC":
        agent = SACAgent(env, wdb)
    elif algorithm == "TD3":
        agent = TD3Agent(env, wdb)
    else:
        raise ValueError("Please, select a valid algorithm! [PPO Continuous, SAC, TD3]")

    print("-" * 20)
    print(f"Training {algorithm} agent on {name} environment.")
    print(f"Agent type is {type(agent)}.")
    print(f"Logging is set to {args.log}.")
    print(f"Hyperparameters: {wdb.hyperparameters}")
    print("-" * 20)

    agent.train()
    env.close()
    wdb.finish()
