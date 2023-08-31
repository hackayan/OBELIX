import argparse
import cv2

import numpy as np

from obelix import OBELIX


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--scaling_factor", help="decides the scaling of the bot and the environment", type=int, default=5)
    args = parser.parse_args()
    
    bot = OBELIX(scaling_factor=5)
    move_choice = ['L45', 'L22', 'FW', 'R22', 'R45']
    user_input_choice = [ord("q"), ord("a"), ord("w"), ord("d"), ord("e")]
    bot.render_frame()
    episode_reward = 0
    for step in range(1, 2000):
        # random_step = np.random.choice(user_input_choice, 1, p=[0.05, 0.1, 0.7, 0.1, 0.05])[0]
        # # random_step = np.random.choice(user_input_choice, 1, p=[0.2, 0.2, 0.2, 0.2, 0.2])[0]
        # x = random_step
        x = cv2.waitKey(0)
        if x in user_input_choice:
            x = move_choice[user_input_choice.index(x)]
            sensor_feedback, reward, done = bot.step(x)
            episode_reward += reward
            print(step, sensor_feedback, episode_reward)
    cv2.waitKey(0)
    exit()
