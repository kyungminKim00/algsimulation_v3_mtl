import gym
from custom_model.policies.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from rllearn.common.vec_env import SubprocVecEnv
from rllearn.common import set_global_seeds
from custom_model.fund_selection.acer.acer_simple import ACER
import util
import my_registration as rt

class script:
    def __init__(self):
        None

    def run(self):
        # multiprocess environment
        n_cpu = 1

        #option1. Direct mapping
        #env = SubprocVecEnv([lambda: rt.registry.make('Test-v0', entry_point='envs.cartpole:CartPoleEnv') for i in range(n_cpu)])

        #option2. use api 
        env = SubprocVecEnv([lambda: util.make('Test-v0') for i in range(n_cpu)])


        model = ACER(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=25000)
        model.save("./save/model/acer_cartpole")
        print('model is saved ')

        del model # remove to demonstrate saving and loading

        print('loading model ')
        model = ACER.load("./save/model/acer_cartpole")
        print('done...!!!')

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

if __name__=='__main__':
    sc = script()
    sc.run()
