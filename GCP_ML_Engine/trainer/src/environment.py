from sklearn.preprocessing import OneHotEncoder

import numpy as np
import tensorflow as tf

from trainer.src.util import python_shuffle


class Environment:
    def __init__(self):
        self.enc = OneHotEncoder(n_values=tf.app.flags.FLAGS.number_goal_types, sparse=False)
        self.colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        self.cols, self.cols_agents, self.cols_landmarks = self.create_colors()

    def create_colors(self):
        cols_agents = np.concatenate(
            [np.tile(np.reshape(self.colors[i], [1, tf.app.flags.FLAGS.color_size, 1]), [1, 1, tf.app.flags.FLAGS.batch_size])
             for i in range(tf.app.flags.FLAGS.number_agents)], axis=0)
        cols_landmarks = np.concatenate(
            [np.tile(np.reshape(self.colors[i], [1, tf.app.flags.FLAGS.color_size, 1]), [1, 1, tf.app.flags.FLAGS.batch_size])
             for i in range(tf.app.flags.FLAGS.number_landmarks)], axis=0)

        cols = np.concatenate([cols_agents, cols_landmarks], axis=0)

        return cols, cols_agents, cols_landmarks

    def create_consistent_targets(self):
        targets_by_exp = [np.random.choice(tf.app.flags.FLAGS.number_agents, (tf.app.flags.FLAGS.number_agents, 1), replace=False) for _ in
                          range(tf.app.flags.FLAGS.batch_size)]
        # targets_by_exp = [np.array([[0], [1]]) for _ in range(FLAGS.batch_size)]
        targets_batch = np.stack(targets_by_exp, axis=2)
        return targets_batch

    def create_goal_locations(self, pos_landmarks):
        landmark_nb = [np.random.choice(tf.app.flags.FLAGS.number_landmarks, (tf.app.flags.FLAGS.number_agents, 1), replace=True) for _ in
                       range(tf.app.flags.FLAGS.batch_size)]
        landmark_nb_batch = np.stack(landmark_nb, axis=2)

        goal_loc = python_shuffle(pos_landmarks, landmark_nb_batch)

        return goal_loc

    def random_generation(self):
        positions_agents = np.random.uniform(-tf.app.flags.FLAGS.bound, tf.app.flags.FLAGS.bound, (tf.app.flags.FLAGS.number_agents,
                                                                                                   tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size))

        # positions_agents = np.array([[[-1 for i in range(100)], [-1 for i in range(100)]], [[1 for i in range(100)], [1 for i in range(100)]]])

        # positions_landmarks = np.random.uniform(-FLAGS.bound, FLAGS.bound, (FLAGS.number_landmarks,
        #                                                          FLAGS.dim_env, FLAGS.batch_size))
        positions_landmarks = np.array(
            [[[-4 for i in range(100)], [-4 for i in range(100)]], [[4 for i in range(100)], [4 for i in range(100)]]])
        positions = np.concatenate([positions_agents, positions_landmarks], axis=0)

        gazes = np.random.uniform(-tf.app.flags.FLAGS.bound, tf.app.flags.FLAGS.bound, (tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks,
                                                                                        tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size))

        # velocities = np.random.uniform(-FLAGS.bound, FLAGS.bound, (FLAGS.number_agents + FLAGS.number_landmarks,
        #                                                          FLAGS.dim_env, FLAGS.batch_size))

        velocities = np.zeros([tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, tf.app.flags.FLAGS.dim_env, tf.app.flags.FLAGS.batch_size])

        # goal_locations = np.random.uniform(-FLAGS.bound, FLAGS.bound, [FLAGS.number_agents,
        #                                                          FLAGS.dim_env, FLAGS.batch_size])

        goal_locations = self.create_goal_locations(positions_landmarks)

        goal_types = np.concatenate([np.reshape(np.transpose(self.enc.fit_transform(
            np.random.choice(tf.app.flags.FLAGS.number_goal_types, tf.app.flags.FLAGS.batch_size).reshape(-1, 1))),
            [1, tf.app.flags.FLAGS.number_goal_types, tf.app.flags.FLAGS.batch_size]) for _ in range(tf.app.flags.FLAGS.number_agents)], axis=0)

        # goal_types = np.array([[[0 for i in range(FLAGS.batch_size)], [1 for i in range(FLAGS.batch_size)], [0 for i in range(FLAGS.batch_size)]]])
        utterances = np.zeros((tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.vocabulary_size, tf.app.flags.FLAGS.batch_size))
        memories_com = np.zeros((tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.mem_size, tf.app.flags.FLAGS.batch_size))
        memories_last = np.zeros((tf.app.flags.FLAGS.number_agents, tf.app.flags.FLAGS.last_mem_size, tf.app.flags.FLAGS.batch_size))

        states = np.concatenate([positions, velocities, gazes, self.cols], axis=1)
        targets = self.create_consistent_targets()

        return states, utterances, memories_com, memories_last, goal_locations, goal_types, targets