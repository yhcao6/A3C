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

        # optimizer
        self.actor_optimiizer = self.actor_optimiizer()
        self.critic_optimizer = self.critic_optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary', self.sess.graph)

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

    # input is state, action_one_hot, advantage
    # optimize actor
    def actor_optimiizer(self):
        action_one_hot = K.placeholder(shape=[None, ACTION_SIZE])
        advantages = K.placeholder(shape=[None,])

        policy = self.actor.output
        p_a_pi = K.sum(policy * action_one_hot, axis=1)
        actor_loss = -K.mean(advantages * K.log(p_a_pi + 1e-10))

        entropy = K.sum(policy * -K.log(policy + 1e-10), axis=1)
        entropy_loss = K.mean(entropy)

        loss = actor_loss + 0.01 * entropy_loss
        optimizer = RMSprop(lr=ACTOR_LR, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action_one_hot, advantages], [loss], updates=updates)
        return train

    # input is state, target value
    # optimize critic
    def critic_optimizer(self):
        target_values = K.placeholder(shape=[None,])
        values = self.critic.output
        loss = K.mean(K.square(target_values - values))  # MSE
        optimizer = RMSprop(lr=CRITIC_LR, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target_values], [loss], updates=updates)
        return train

    def setup_summary(self):
        total_reward = tf.Variable(0.)
        duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', total_reward)
        tf.summary.scalar('Duration/Episode', duration)

        summary_vars = [total_reward, duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


class Agent:
    def __init__(self):
        self.histories = []  # store experience in one episode
        self.action_one_hots = []
        self.rewards = []

    def get_epsilon(self):
        if frames >= EPS_STEPS:
            return EPS_STOP
        else:
            return EPS_START - (EPS_START - EPS_STOP) / EPS_STEPS * frames

    # e-greedy
    def choose_action(self, history):
        global frames
        frames += 1
        eps_current = self.get_epsilon()

        if random.random() < eps_current:
            return random.randint(0, ACTION_SIZE - 1)
        else:
            policy = brain.actor.predict(history)[0]
            return np.random.choice(ACTION_SIZE, 1, p=policy)

    def observe(self, history, action, reward):
        self.histories.append(history)
        action_one_hot = np.zeros(ACTION_SIZE)
        action_one_hot[action] = 1
        self.action_one_hots.append(action_one_hot)
        self.rewards.append(reward)

    def get_targets(self, rewards, done):
        targets = np.zeros_like(rewards)
        R = 0.
        if not done:
            R = brain.critic.predict(self.histories[-1])[0]
            targets[-1] = 0
        else:
            R = self.rewards[-1]
            targets[-1] = R
        for t in reversed(range(0, len(rewards) - 1)):
            R = R * DISCOUNT_FACTOR + rewards[t]
            targets[t] = R
        return targets

    def train(self, done):
        targets = self.get_targets(self.rewards, done)
        histories = np.array(self.histories)[:, 0, ...]  # 20 x 84 x 84 x 4

        values = brain.critic.predict(histories)
        values = np.reshape(values, len(values))

        advantages = targets - values

        brain.actor_optimiizer([histories, self.action_one_hots, advantages])
        brain.critic_optimizer([histories, targets])
        self.histories, self.action_one_hots, self.rewards = [], [], []


class Environment(threading.Thread):
    stop_signal = False

    def __init__(self):
        # every environment runs in an environment
        threading.Thread.__init__(self)

        self.env = gym.make(PROBLEM).unwrapped
        # every environment contains an agent
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
            step += 1
            t += 1

            a = self.agent.choose_action(history)

            # reduce one action, accelerate training
            # fire is same as static except first start
            if a == 0:
                real_a = 1
            elif a == 1:
                real_a = 2
            else:
                real_a = 3

            # if dead, restart the game
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
            r = np.clip(r, -1., 1.)  # clip reward

            self.agent.observe(history, a, r)

            # if dead, then history should only related start state
            if dead:
                s_ = s_[0, ..., 0]
                history = np.stack((s_, s_, s_, s_), axis=2)  # 84 x 84 x 4
                history = history[np.newaxis, ...]  # 1 x 84 x 84 x 4
            else:
                history = history_

            if t >= T_MAX or done:
                self.agent.train(done)
                t = 0

            if done:
                print 'At episode', episodes, 'score is', score

                states = [score, step]
                for i in range(len(states)):
                    brain.sess.run(brain.update_ops[i], feed_dict={brain.summary_placeholders[i]: float(states[i])})
                summary_str = brain.sess.run(brain.summary_op)
                brain.summary_writer.add_summary(summary_str, episodes)

        if episodes % 1000 == 0:
            brain.actor.save_weights('actor_' + str(episodes) + '.h5')
            brain.critic.save_weights('critic_' + str(episodes) + '.h5')

        if episodes > EPISODES:
            self.stop_signal = True

    def run(self):
        while not self.stop_signal:
            self.run_episode()


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
envs = []
for _ in range(NUM_THREADS):
    envs.append(Environment())

for env in envs:
    time.sleep(1)
    env.start()
