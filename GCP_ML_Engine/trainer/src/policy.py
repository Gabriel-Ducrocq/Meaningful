class Policy:
    def __init__(self):
        self.phys_network = PhysicalNet()
        self.comm_network = CommunicationNet()
        self.last_network = LastNet()

        self.define_placeholders()
        self.define_full_goals()

    def define_placeholders(self):
        self.states = tf.placeholder(tf.float32, [FLAGS.number_agents + FLAGS.number_landmarks,
                                                  3 * FLAGS.dim_env + FLAGS.color_size, FLAGS.batch_size])
        self.utterances = tf.placeholder(tf.float32, [FLAGS.number_agents, FLAGS.vocabulary_size, FLAGS.batch_size])
        self.memories_com = tf.placeholder(tf.float32, [FLAGS.number_agents, FLAGS.mem_size, FLAGS.batch_size])
        self.memories_last = tf.placeholder(tf.float32, [FLAGS.number_agents, FLAGS.last_mem_size, FLAGS.batch_size])
        self.goal_types = tf.placeholder(tf.float32, [FLAGS.number_agents, FLAGS.number_goal_types, FLAGS.batch_size])
        self.goal_locations = tf.placeholder(tf.float32, [FLAGS.number_agents, FLAGS.dim_env, FLAGS.batch_size])
        self.name_targets = tf.placeholder(tf.int32, [FLAGS.number_agents, 1, FLAGS.batch_size])
        # self.colors = tf.placeholder(tf.float32, [FLAGS.number_agents, FLAGS.color_size, FLAGS.batch_size])

    def define_full_goals(self):
        colors = tf.slice(self.states, [0, 3 * FLAGS.dim_env, 0],
                          [FLAGS.number_agents, FLAGS.color_size, FLAGS.batch_size])
        shuffled_colors = shuffle(colors, self.name_targets, colors=True)
        self.full_goals = tf.concat([self.goal_types, self.goal_locations, shuffled_colors], axis=1)

    def get_placeholders(self):
        return [self.states, self.utterances, self.memories_com, self.memories_last, self.goal_types,
                self.goal_locations,
                self.full_goals, self.name_targets]

    def forward_pass(self, states, utterances, mem, mem_last, goals_last):
        # Step 1: processing observed states and utterances
        phys_output = self.phys_network.compute_output(states)
        comm_output, delta_mem_com = self.comm_network.compute_output(utterances, mem)

        # Step 2: softmax pooling the results [num_agents, output size, batch_size] --> [1, output size, batch_size]
        PhiX = softmax_pooling(phys_output)
        PhiC = softmax_pooling(comm_output)

        # Step 3: feeding the last network
        PhiX_last = tf.tile(tf.reshape(PhiX, [1, FLAGS.output_size, FLAGS.batch_size]), [FLAGS.number_agents, 1, 1])
        PhiC_last = tf.tile(tf.reshape(PhiC, [1, FLAGS.output_size, FLAGS.batch_size]), [FLAGS.number_agents, 1, 1])

        input_last = tf.concat([PhiX_last, goals_last, PhiC_last], axis=1)

        output_last, delta_mem_last = self.last_network.compute_output(input_last, mem_last)

        velocities_output, gazes_output = sample_phys(output_last)
        utterances_output = gumbel_max_trick(output_last)
        phys_output = tf.concat([velocities_output, gazes_output], axis=1)

        return phys_output, velocities_output, gazes_output, utterances_output, delta_mem_com, delta_mem_last