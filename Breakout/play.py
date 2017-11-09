import numpy as np
import tensorflow as tf
import threading
import random
import time
import gym
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from skimage.color import rgb2gray
from skimage.transform import resize


# configuration
PROBLEM = "BreakoutDeterministic-v4"
STATE_SIZE = (84, 84, 4)
ACTION_SIZE = 3
DISCOUNT_FACTOR = 0.99
NO_OP_MAX = 30
ACTOR_LR = 2.5e-4
CRITIC_LR = 2.5e-4
NUM_THREADS = 8
NUM_OPTIMIZERS = 2
EPS_START = 1.0
EPS_STOP = 0.1
EPS_STEPS = 30000
T_MAX = 20
EPISODES = 8000000


# all agents share same brain
class Brain:
    def __init__(self):
        # two branches, actor, critic
        self.actor, self.critic = self._build_model()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        input = Input(shape=(STATE_SIZE))
        conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv2 = Flatten()(conv2)
        dense = Dense(256, activation='relu')(conv2)
        policy = Dense(ACTION_SIZE, activation='softmax')(dense)
        value = Dense(1, activation='linear')(dense)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # to prevent delay when first initialization
        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic


class Agent:
    def __init__(self):
        pass

    def choose_action(self, history):
        policy = brain.actor.predict(history)[0]
        return np.random.choice(ACTION_SIZE, 1, p=policy)

class Environment:
    def __init__(self):
        self.env = gym.make(PROBLEM).unwrapped
        self.agent = Agent()

    def run_episode(self):
        global episodes
        episodes += 1

        s = self.env.reset()
        done = False
        dead = False
        step = 0
        t = 0
        score, start_life = 0, 5

        # random start
        for _ in range(random.randint(1, NO_OP_MAX)):
            s, _, _, _ = self.env.step(1)

        s = preprocess(s)  # 84 x 84
        history = np.stack((s, s, s, s), axis=2)  # 84 x 84 x 4
        history = history[np.newaxis, ...]  # 1 x 84 x 84 x 4 (N x H x W x C)

        while not done:
            self.env.render()
            a = self.agent.choose_action(history)

            if a == 0:
                real_a = 1
            elif a == 1:
                real_a = 2
            else:
                real_a = 3

            if dead:
                a = 0
                real_a = 1
                dead = False

            s_, r, done, info = self.env.step(real_a)

            s_ = preprocess(s_)  # 84 x 84
            s_ = s_[np.newaxis, ..., np.newaxis]  # 1 x 84 x 84 x 1
            history_ = np.append(s_, history[..., :3], axis=3)  # 1 x 84 x 84 x 4

            # if lives decrease, then agent is dead
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += r  # real score

            # if dead, then history should only related start state
            if dead:
                s_ = s_[0, ..., 0]
                history = np.stack((s_, s_, s_, s_), axis=2)  # 84 x 84 x 4
                history = history[np.newaxis, ...]  # 1 x 84 x 84 x 4
            else:
                history = history_

            if done:
                print 'At episode', episodes, 'score is', score
                break

def preprocess(s):
    # from rgb to gray
    gray = rgb2gray(s)
    # resize to STATE_SIZE by bilinear interpolation
    gray_resize = resize(gray, (84, 84), mode='constant')
    return gray_resize


# global variable share by threads
episodes = 0
frames = 0

brain = Brain()
brain.actor.load_weights('actor_1000.h5')
brain.critic.load_weights('critic_1000.h5')
env = Environment()
env.run_episode()
