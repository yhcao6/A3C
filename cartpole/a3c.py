import gym
import time
import random
import threading
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import RMSprop


ENV = 'CartPole-v0'

THREADS = 8
OPTIMIZERS = 2

GAMMA = 0.99

N_STEP_RETURN = 9
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
ACTOR_LR = 5e-3
CRITIC_LR = 5e-3

LOSS_V = .5
LOSS_ENTROPY = .01

STATE_SIZE = 4
ACTION_SIZE = 2

RUN_TIME = 40


NONE_STATE = np.zeros(STATE_SIZE)
global_steps = 0
global_episodes = 0

class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', terminal
    queue_lock = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.actor, self.critic = self._build_model()
        self.target_actor, self.target_critic = self._build_model()
        self.actor_optimizer = self.actor_optimizer()
        self.critic_optimizer = self.critic_optimizer()

        self.session.run(tf.global_variables_initializer())
        self.update()

    def _build_model(self):
        input = Input(batch_shape=(None, STATE_SIZE))
        dense = Dense(16, activation='relu')(input)
        policy = Dense(ACTION_SIZE, activation='softmax')(dense)
        value = Dense(1, activation='linear')(dense)

        actor = Model(inputs=[input], outputs=[policy])
        critic = Model(inputs=[input], outputs=[value])

        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic

    def update(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def train_push(self, s, a, r, s_):
        with self.queue_lock:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            return

        with self.queue_lock:
            if len(self.train_queue[0]) < MIN_BATCH:
                return

            s, a, r, s_, terminal = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        terminal = np.vstack(terminal)

        v_s_ = self.target_critic.predict(s_)
        target_value = r + GAMMA_N * v_s_ * terminal
        advantage = target_value - self.critic.predict(s)
        self.actor_optimizer([s, a, advantage])
        self.critic_optimizer([s, target_value])

    def actor_optimizer(self):
        action_mask = K.placeholder(shape=[None, ACTION_SIZE])
        advantage = K.placeholder(shape=[None, 1])

        policy = self.actor.output
        choose_policy = K.sum(policy * action_mask, axis=1)
        log_prob = K.log(choose_policy + 1e-10)
        loss_policy = -K.sum(log_prob * advantage)

        loss_entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        loss_entropy = K.sum(loss_entropy)

        loss = loss_policy + 0.01 * loss_entropy
        optimizer = RMSprop(lr=ACTOR_LR, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action_mask, advantage], [loss], updates=updates)
        return train

    def critic_optimizer(self):
        target_value = K.placeholder(shape=[None, 1])
        value = self.critic.output
        loss = K.mean(K.square(target_value - value))

        optimizer = RMSprop(lr=ACTOR_LR, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.actor.input, target_value], [loss], updates=updates)
        return train

    def save_model(self):
        self.actor.save_weights('actor.h5')
        self.critic.save_weights('critic.h5')


class Agent:
    def __init__(self):
        self.memory = []
        self.R = 0.

    def get_epsilon(self):
        if global_steps >= EPS_STEPS:
            return EPS_STOP
        else:
            return EPS_START - global_steps * (EPS_START - EPS_STOP) / EPS_STEPS


    def choose_action(self, s):
        global global_steps
        global_steps += 1
        if global_steps % 7500 == 0:
            brain.update()
        epsilon = self.get_epsilon()
        if random.random() < epsilon:
            a = random.randint(0, ACTION_SIZE-1)
        else:
            policy = brain.actor.predict(s)
            a = np.random.choice(ACTION_SIZE, p=policy[0])

        return a

    def observe(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_one_hot = np.zeros(ACTION_SIZE)
        a_one_hot[a] = 1.

        self.memory.append((s, a_one_hot, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, id):
        threading.Thread.__init__(self)
        self.id = id
        self.env = gym.make(ENV)
        self.agent = Agent()

    def run_episode(self, render):
        global global_episodes
        global_episodes += 1
        s = self.env.reset()
        s = s[np.newaxis, ...]
        R = 0

        while True:
            if render:
                self.env.render()
            a = self.agent.choose_action(s)

            s_, r, done, info = self.env.step(a)
            s_ = s_[np.newaxis, ...]

            if done:
                s_ = None

            self.agent.observe(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print 'thread', self.id, 'episode', global_episodes, 'steps', global_steps, 'total reward:', R


    def run(self, render=False):
        while not self.stop_signal:
            self.run_episode(render)

    def stop(self):
        self.stop_signal = True


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self, id):
        threading.Thread.__init__(self)
        self.id = id

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


brain = Brain()

envs = []
for i in range(THREADS):
    envs.append(Environment(i))

opts_id = [0, 1]
opts = []
for i in opts_id:
    opts.append(Optimizer(i))

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()

for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

brain.save_model()
EPS_START = 0
EPS_STOP = 0
env = Environment(1)
env.run(render=True)




