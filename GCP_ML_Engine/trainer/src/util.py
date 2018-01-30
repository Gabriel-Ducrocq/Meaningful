import tensorflow as tf
import numpy as np


# Param: x, stacking of the output of fully connected physical network for each agent. Shape = (256, batch_size, nb_agents)
# return: pooling of input features.


def softmax_pooling(x):
    # pooling function. Softmax pooling is a compromise between max pooling and average pooling
    coefs = tf.nn.softmax(x, dim=0)
    softmax_pool = tf.reduce_sum(tf.multiply(coefs, x), axis=0)
    return softmax_pool


def activation_function(x):
    return tf.nn.elu(x)


def gumbel_max_trick(x):
    # Application of gumbel-softmax trick
    # Input: output of the last network
    u = -tf.log(-tf.log(tf.random_uniform(shape=[tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.vocabulary_size, tf.app.flags.FLAGS.batch_size],
                                          dtype=tf.float32)))
    utterance_output = tf.slice(x, [0, 2 * tf.app.flags.FLAGS.dim_env, 0],
                                [tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.vocabulary_size, tf.app.flags.FLAGS.batch_size])
    gumbel = tf.exp((utterance_output + u) / tf.app.flags.FLAGS.gumbel_temperature)
    denoms = tf.reshape(tf.reduce_sum(gumbel, axis=1), [tf.app.flags.FLAGS.number_agents, 1, tf.app.flags.FLAGS.batch_size])
    utterance = gumbel / denoms
    return utterance


def sample_phys(x):
    # Input output of the last network.
    # Output: sampled values for new velocity and gaze
    u = tf.random_normal(shape=[tf.app.flags.FLAGS.number_agents, 2 * tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size], dtype=tf.float32,
                         stddev=tf.app.flags.FLAGS.sddev_phys_sampling)
    o = tf.add(tf.slice(x, [0, 0, 0], [tf.app.flags.FLAGS.number_agents, 2 * tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size]), u)
    sample_move = tf.slice(o, [0, 0, 0], [tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size])
    sample_gaze = tf.slice(o, [0, tf.app.flags.FLAGS.dim_env, 0], [tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size])
    return sample_move, sample_gaze


def compute_new_states(old_states, new_velocities, new_gazes, new_utterances):
    # Computes the new states according to the equations of the papers.
    # Input: the old states of shape [number agents + nb_landmarks, 3*env dim + color size, batch size] because color is in state
    # and of shape [number_agents, 2*env_dim, batch size]
    # Adding the outputs of landmark, which are all zeros.
    new_velocities = tf.concat([new_velocities, tf.zeros([tf.app.flags.FLAGS.number_landmarks, tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size])],
                               axis=0)
    new_gazes = tf.concat([new_gazes, tf.zeros([tf.app.flags.FLAGS.number_landmarks, tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size])],
                          axis=0)

    old_velocity = tf.slice(old_states, [0, tf.app.flags.FLAGS.dim_env, 0],
                            [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size])
    new_pos = tf.slice(old_states, [0, 0, 0],
                       [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size]) + old_velocity

    new_velocity = (1 - tf.app.flags.FLAGS.damping_coef) * old_velocity + new_velocities * tf.app.flags.FLAGS.delta_t

    colors = tf.slice(old_states, [0, 3 * tf.app.flags.FLAGS.dim_env, 0], [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks,
                                                                           tf.app.flags.FLAGS.color_size, tf.app.flags.FLAGS.batch_size])
    new_states = tf.concat([new_pos, new_velocity, new_gazes, colors], axis=1)

    return new_states, new_pos


def compute_new_memories(old_mem_com, old_mem_last, delta_mem_com, delta_mem_last):
    new_memory_com = tf.tanh(old_mem_com + delta_mem_com + tf.random_normal([tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.mem_size,
                                                                             tf.app.flags.FLAGS.batch_size], tf.app.flags.FLAGS.stddev_memory))
    new_memory_last = tf.tanh(old_mem_last + delta_mem_last + tf.random_normal([tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.mem_size,
                                                                                tf.app.flags.FLAGS.batch_size], tf.app.flags.FLAGS.stddev_memory))

    return new_memory_com, new_memory_last


def shuffle(x, name_targets, colors=False):
    slices_second_dim = []
    ones = tf.ones([tf.app.flags.FLAGS.number_agents, 1, tf.app.flags.FLAGS.batch_size], tf.int32)
    batch_num = tf.tile(tf.reshape(tf.range(0, tf.app.flags.FLAGS.batch_size, dtype=tf.int32), [1, 1, tf.app.flags.FLAGS.batch_size]),
                        [tf.app.flags.FLAGS.number_agents,
                         1, 1])
    if not colors:
        for i in range(tf.app.flags.FLAGS.dim_env):
            slices_second_dim.append(tf.reshape(tf.concat([name_targets, ones * i, batch_num], axis=1),
                                                [tf.app.flags.FLAGS.number_agents, 1, 3, tf.app.flags.FLAGS.batch_size]))
    else:
        for i in range(tf.app.flags.FLAGS.color_size):
            slices_second_dim.append(tf.reshape(tf.concat([name_targets, ones * i, batch_num], axis=1),
                                                [tf.app.flags.FLAGS.number_agents, 1, 3, tf.app.flags.FLAGS.batch_size]))

    gathering_tensor = tf.transpose(tf.concat(slices_second_dim, axis=1), perm=[0, 1, 3, 2])
    shuffled_x = tf.gather_nd(x, gathering_tensor)

    return shuffled_x


def compute_reward(positions, gazes, outputs, utterances, name_targets, goals_loc, goals_types):
    shuffled_positions = shuffle(positions, name_targets)
    shuffled_gazes = shuffle(gazes, name_targets)

    pos_distances = tf.reshape(tf.reduce_sum(tf.square((shuffled_positions - goals_loc)), axis=1),
                               [tf.app.flags.FLAGS.number_agents, 1,
                                tf.app.flags.FLAGS.batch_size])
    gaze_distances = tf.reshape(tf.reduce_sum(tf.square((shuffled_gazes - goals_loc)), axis=1), [tf.app.flags.FLAGS.number_agents, 1,
                                                                                                 tf.app.flags.FLAGS.batch_size])
    zeros = tf.zeros([tf.app.flags.FLAGS.number_agents, 1, tf.app.flags.FLAGS.batch_size])
    x = tf.concat([pos_distances, gaze_distances, zeros], axis=1)
    dists_goal = -tf.reduce_sum(tf.multiply(x, goals_types), axis=1)

    utterances_term = -tf.reduce_sum(tf.square(utterances), axis=1)
    output_term = -tf.reduce_sum(tf.square(outputs), axis=1)

    reward_by_batch = tf.reshape(tf.reduce_sum(dists_goal + utterances_term + 0.1 * output_term, axis=0),
                                 [tf.app.flags.FLAGS.batch_size, 1])

    return reward_by_batch


def compute_goal_dist(states, goal_location, goal_type):
    dist_positions = np.reshape(np.sqrt(np.sum((states[0:tf.app.flags.FLAGS.number_agents, 0:2, :] - goal_location) ** 2, axis=1)),
                                [tf.app.flags.FLAGS.number_agents, 1, tf.app.flags.FLAGS.batch_size])
    dist_gazes = np.reshape(np.sqrt(np.sum((states[0:tf.app.flags.FLAGS.number_agents, 4:6, :] - goal_location) ** 2, axis=1)),
                            [tf.app.flags.FLAGS.number_agents, 1, tf.app.flags.FLAGS.batch_size])

    v = np.concatenate([dist_positions, dist_gazes, np.zeros((tf.app.flags.FLAGS.number_agents, 1, tf.app.flags.FLAGS.batch_size))], axis=1)
    goal_distances = np.sum(np.multiply(v, goal_type), axis=1)

    return goal_distances


def print_stats_agent(states, goal_location, goal_type):
    # Only considering non "do nothing goals"
    goal_distances = compute_goal_dist(states, goal_location, goal_type)

    for i in range(tf.app.flags.FLAGS.number_agents):
        distances_agents = goal_distances[i, :]
        goal_wo_zeros = distances_agents[distances_agents != 0]
        mean = np.mean(goal_wo_zeros)
        median = np.median(goal_wo_zeros)
        third_quart = np.percentile(goal_wo_zeros, 75)
        nine_pct = np.percentile(goal_wo_zeros, 90)
        max_dist = np.max(goal_wo_zeros)
        print("--- Agent " + str(i))
        print("------ Mean distance " + str(mean))
        print("------ Median distance " + str(median))
        print("------ Third quartile " + str(third_quart))
        print("------ Ninetieth percentile " + str(nine_pct))
        print("------ max distance " + str(max_dist))