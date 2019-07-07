import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import gym

mspacman_color = 210 + 164 + 74  # WHY?


# to make this script's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def preprocess_observation(obs):
    """ Reduce size of input images, make greyscale, improve contrast,
    etc. Will speed up learning. """
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to grayscale
    img[img == mspacman_color] = 0  # improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)

"""
def plot_state(obs):
    dims = obs.shape
    y = np.arange(dims[0])
    x = np.arange(dims[1])
    plt.contourf(x, y, obs[::-1, :, 0], cmap='Greys', vmin=-1., vmax=1.)
    plt.show()
"""


env = gym.make("MsPacman-v0")
obs = env.reset()

"""
n_steps = 90  # waiting phase ...
for i in range(n_steps):
    obs, r, done, info = env.step(env.action_space.sample())

fixed_action = 0
for i in range(10):
    obs, r, done, info = env.step(fixed_action)
    obs = preprocess_observation(obs)
    plot_state(obs)
"""

# (I) CONSTRUCTION PHASE
# Define DQN:
# (1) should output approximative function Q(s,a). Instead of
# actually taking (s, a) as input parameters, we limit ourselves to
# input parameter (s), and output a vector of Q values (one entry for
# each action). This is a better approach since the actions are anyway
# discrete.
# (2) We will actually define 2 DQNs with the same architecture, but
# different weights (like DeepMind did): an 'online' and a 'target' NN.
# Every so iterations the two NN will be synced.
# (3) We will only use one frame, since there is almost no hidden state,
# except some blinking stuff and the directions of the ghosts. In other
# games, such as Pong or Breakout, we'd need to include several frames
# in our state, since we'd be dealing with a moving ball: direction and
# speed can only be derived from several (at least 2) frames.
reset_graph()

input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]  # first 3 layers: CNNs
conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
conv_strides = [4, 2, 1]  # strides for kernel shifts
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11*10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # one q value per discrete action
initializer = tf.variance_scaling_initializer()


def q_network(X_state, name):
    """ trainable_vars_by_name: contains all the trainable variables
    of this DQN. Will be useful when we create operations to copy the
    online DQN onto the target DQN. """
    # scale pixel intensities to the [-1.0, 1.0] range.
    prev_layer = X_state / 128.
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(
            prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var
                                  for var in trainable_vars}
        return outputs, trainable_vars_by_name


X_state = tf.placeholder(
    tf.float32, shape=[None, input_height, input_width, input_channels])

# Define 2 DQNs capable of taking input state (X_state: a single
# preprocessed observation / image) as input, and outputting an estim.
# Q-value for each possible action in that state.
online_q_values, online_vars = q_network(
    X_state, name="q_networks/online")
target_q_values, target_vars = q_network(
    X_state, name="q_networks/target")

# Define copy_online_to_target to copy the values of all the trainable
# variables of the online DQN to the corresponding target DQN variables.
# TensorFlow's tf.group(..) function groups all assignment operations
# into a single, convenient operation.
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

learning_rate = 0.001
momentum = 0.95
with tf.variable_scope("train"):
    # Add online DQN's training operations. Need to be able to compute
    # Q-value for each state-action pair in the memory batch. DQN outputs
    # one Q-value for every action => keep only Q-value for the action that
    # was actually played => convert action to a one-hot vector and multiply
    # it by the Q-values (output from DQN), then sum over the first axis to
    # get the Q-value corresponding to the played action for each memory.
    # (So this is already taking into account that there will be experience
    # replay).

    # Also create placeholder y that we use to provide target Q-values,
    # and compute the loss: we use squared error when error < 1.0 and
    # 2*abs error when error > 1.0 => loss is quadratic for small erorrs
    # and linear for large errors => reduces effect of large errors and
    # helps stabilise training.
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(
        online_q_values * tf.one_hot(X_action, n_outputs), axis=1,
        keep_dims=True)

    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0., 1.)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    # Create Nesterov Accelerated Gradient (NAG) optimiser to minimise
    # loss.
    # global_step keeps track of number of training steps: training_op
    # will take care of that.
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# (II) TOOLS
# Experience replay: implement replay memory using a user-defined class
# and small function to randomly sample batch from memory. Each memory,
# or, experience will be a 5-tuple: (state, action, reward, next state,
# continue). 'continue' is either 0 (game over), or 1 (game continues).
class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)  # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]


replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)


def sample_memories(batch_size):
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],\
           cols[4].reshape(-1, 1)


# (III) POLICY
# Agent needs to explore the game: we use the epsilon-greedy policy.
# Linearly decrease epsilon from 1.0 to 0.1 in 2e6 steps.
eps_min = 0.1
eps_max = 1.0
eps_decay_steps = int(2e6)

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min,
                  eps_max - (eps_max - eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)  # take random action
    else:
        return np.argmax(q_values)  # optimal action (highest Q-value)


# (IV) EXECUTION PHASE
n_steps = int(4e6)  # total number of training steps
training_start = int(1e4)  # start training after 10.000 game iterations
training_interval = 4  # run training step every 4 game iterations
save_steps = 1000  # save the model every 1.000 training steps
copy_steps = int(1e4)  # copy online DQN to target DQN every 10.000 training steps
discount_rate = 0.99  # future reward discount factor
skip_start = 90  # Skip the start of every game (just waiting time)
batch_size = 50  # for experience replay?
iteration = 0  # game iterations
checkpoint_path = './my_dqn.ckpt'  # where to save DQNs
done = True  # env needs to be reset

# Some variables to track progress
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

# Open TF session and run main training loop
with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()

    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        print("\rIteration {}\tTraining step {}/{} ({:.1f})%\t"
              "Loss {:5f}\tMean Max-Q {:5f}   ".format(
            iteration, step, n_steps, step * 100 / n_steps,
            loss_val, mean_max_q), end="")
        if done:  # game over, start again
            obs = env.reset()
            for skip in range(skip_start):  # skip start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Let's memorise what just happened and move on
        replay_memory.append(
            (state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Compute statistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        # only train after warmup period and at regular intervals
        if ((iteration < training_start) or
                (iteration % training_interval != 0)):
            continue

        # Sample memories and use target DQN to produce target Q-value
        # Future reward part for target Q-values for current state are
        # given by the next_state evaluated on target DQN
        # Immediate reward on target is given by the reward that was
        # received when performing that step
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val = sess.run(
            [training_op, loss],
            feed_dict={X_state: X_state_val, X_action: X_action_val,
                       y: y_val})

        # Regularly copy online DQN to target DQN (weights)
        if step % copy_steps == 0:
            copy_online_to_target.run()

        # Save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)
            