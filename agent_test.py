# TO TEST THE AGENT
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import gym

mspacman_color = 210 + 164 + 74  # WHY?


def preprocess_observation(obs):
    """ Reduce size of input images, make greyscale, improve contrast,
    etc. Will speed up learning. """
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to grayscale
    img[img == mspacman_color] = 0  # improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


env = gym.make("MsPacman-v0")

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

checkpoint_path = './my_dqn.ckpt'  # from where to restore dqn
saver = tf.train.Saver()

frames = []
n_max_steps = 10000
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    obs = env.reset()
    for step in range(n_max_steps):
        state = preprocess_observation(obs)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = np.argmax(q_values)

        # Online DQN plays
        obs, reward, done, info = env.step(action)

        img = env.render(mode="rgb_array")
        frames.append(img)

        if done:
            break

plt.ion()
fig = plt.figure()
plt.show()
for i in range(len(frames)):
    plt.imshow(frames[i])
    plt.axis('off')
    plt.pause(0.01)
