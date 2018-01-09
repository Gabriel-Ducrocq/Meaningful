class Environment:
    def __init__(self):
        self.enc = OneHotEncoder(n_values=FLAGS.number_goal_types, sparse=False)
        self.colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        self.cols = self.create_colors()

    def create_colors(self):
        cols_agents = np.concatenate(
            [np.tile(np.reshape(self.colors[i], [1, FLAGS.color_size, 1]), [1, 1, FLAGS.batch_size])
             for i in range(FLAGS.number_agents)], axis=0)
        cols_landmarks = np.concatenate(
            [np.tile(np.reshape(self.colors[i], [1, FLAGS.color_size, 1]), [1, 1, FLAGS.batch_size])
             for i in range(FLAGS.number_landmarks)], axis=0)

        cols = np.concatenate([cols_agents, cols_landmarks], axis=0)

        return cols

    def create_consistent_targets(self):
        targets_by_exp = [np.random.choice(FLAGS.number_agents, (FLAGS.number_agents, 1), replace=False) for _ in
                          range(FLAGS.batch_size)]
        targets_batch = np.stack(targets_by_exp, axis=2)
        return targets_batch

    def random_generation(self):
        positions = np.random.uniform(-FLAGS.bound, FLAGS.bound, (FLAGS.number_agents + FLAGS.number_landmarks,
                                                                  FLAGS.dim_env, FLAGS.batch_size))

        gazes = np.random.uniform(-FLAGS.bound, FLAGS.bound, (FLAGS.number_agents + FLAGS.number_landmarks,
                                                              FLAGS.dim_env, FLAGS.batch_size))

        velocities = np.random.uniform(-FLAGS.bound, FLAGS.bound, (FLAGS.number_agents + FLAGS.number_landmarks,
                                                                   FLAGS.dim_env, FLAGS.batch_size))

        goal_locations = np.random.uniform(-FLAGS.bound, FLAGS.bound, [FLAGS.number_agents,
                                                                       FLAGS.dim_env, FLAGS.batch_size])

        goal_types = np.concatenate([np.reshape(np.transpose(self.enc.fit_transform(
            np.random.choice(FLAGS.number_goal_types, FLAGS.batch_size).reshape(-1, 1))),
            [1, FLAGS.number_goal_types, FLAGS.batch_size]) for _ in range(FLAGS.number_agents)], axis=0)

        utterances = np.zeros((FLAGS.number_agents, FLAGS.vocabulary_size, FLAGS.batch_size))
        memories_com = np.zeros((FLAGS.number_agents, FLAGS.mem_size, FLAGS.batch_size))
        memories_last = np.zeros((FLAGS.number_agents, FLAGS.last_mem_size, FLAGS.batch_size))

        states = np.concatenate([positions, velocities, gazes, self.cols], axis=1)
        targets = self.create_consistent_targets()

        return states, utterances, memories_com, memories_last, goal_locations, goal_types, targets