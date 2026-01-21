from src.utils.arg_handler import get_sb3_args
import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == "__main__":
    args = get_sb3_args()

    env = DummyVecEnv([lambda: gym.make(args.env)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    if args.alg == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
    elif args.alg == "SAC":
        model = SAC("MlpPolicy", env, verbose=1)
    elif args.alg == "TD3":
        model = TD3("MlpPolicy", env, verbose=1)
    else:
        print("Use PPO, SAC or TD3")
        exit(1)

    model.learn(total_timesteps=args.steps)
    env.training = False

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.eval,
        deterministic=True
    )

    print("-" * 20)
    print(f"Results for {args.alg} in {args.env}")
    print(f"Mean reward: {mean_reward}")
    print(f"Std reward: {std_reward}")
    print("-" * 20)
