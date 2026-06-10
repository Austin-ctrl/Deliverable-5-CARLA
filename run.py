# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gymnasium as gym
import gym_carla
import carla
from gym_carla.envs.carla_env import CarlaEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 1,
    'number_of_walkers': 0,
    'display_size': 128,
    'max_past_step': 1,
    'dt': 0.1,
    'discrete': False,
    'discrete_acc': [-3.0, 0.0, 3.0],
    'discrete_steer': [-0.2, 0.0, 0.2],
    'continuous_accel_range': [-3.0, 3.0],
    'continuous_steer_range': [-0.3, 0.3],
    'ego_vehicle_filter': 'vehicle.lincoln*',
    'port': 4000,
    'town': 'Town01',
    'max_time_episode': 1000,
    'max_waypt': 12,
    'obs_range': 16,
    'lidar_bin': 0.25,
    'd_behind': 12,
    'out_lane_thres': 2.0,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'display_route': False,
  }

  # Instantiate env directly, bypassing gym.make params issue
  env = CarlaEnv(params)

  model = SAC("MlpPolicy", env, device="cuda", buffer_size=1_000, batch_size=32, learning_starts=500, verbose=1, tensorboard_log="./tensorboard_DQN/")  
  model.learn(total_timesteps=10_000, tb_log_name="SAC_CARLA")
  model.save("SAC_dist")
  del model
  model = SAC.load("SAC_dist")

  obs, info = env.reset()
  
  while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

if __name__ == '__main__':
  main()