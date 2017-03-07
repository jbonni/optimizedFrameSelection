import tensorflow as tf
import time
import numpy as np
import parseConfig
import utils
import importlib
import dataHandler

config = parseConfig.config

env = utils.create_env(config)

tf.logging.set_verbosity(tf.logging.ERROR)
sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)

Agent = getattr(importlib.import_module(
    "agents." + config.agent), config.agent)
agent = Agent(config, sess)
agent.testing(True)

saver = tf.train.Saver(max_to_keep=20)

if config.load_checkpoint != "":
    utils.load_checkpoint(sess, saver, config)
else:
    sess.run(tf.initialize_all_variables())

print("Using agent " + config.agent)
print("On device: " + config.device)

dh = dataHandler()


def generate_dataset():
    for episode in range(config.num_episodes):
        x, r, done, score = env.reset(), 0, False, 0
        ep_begin_t = time.time()
        ep_begin_step_count = agent.step_count
        while not done:
            action = agent.step(x, r)
            dh.addData(x, agnet.Q_np, action, r, done,
                       agent.representations[0], agent.representations[0])
            x, r, done, info = env.step(action)
            score += r
        agent.terminal()
        if episode % 50 == 0:
            print("episode: %i -- score: %i" % (episode, score))

generate_dataset()
