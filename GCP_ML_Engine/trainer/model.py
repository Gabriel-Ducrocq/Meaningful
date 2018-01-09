class Experiment:
    def __init__(self):
        self.policy = Policy()
        self.env = Environment()

        self.get_placeholders()
        self.definition_arrays()
        self.write_arrays()
        self.learning_rate = self.learning_rate_decay()
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.loop()
        self.output_to_run = [self.step, self.array_states_stack, self.array_utterances_stack, self.array_mem_com_stack,
                              self.array_mem_last_stack,
                              self.t_fin, self.reward]
        self.init = tf.global_variables_initializer()

        self.reward_history = []
        self.env_history = []
        self.arrays_history = []

    def learning_rate_decay(self):
        self.global_step = tf.Variable(0, trainable=False)
        if FLAGS.learning_rate_decay:
            starter_learning_rate = FLAGS.learning_rate
            boundaries = [1000]
            values = [FLAGS.learning_rate, FLAGS.learning_rate / 10]
            return tf.train.piecewise_constant(self.global_step, boundaries, values, name=None)
        else:
            return FLAGS.learning_rate

    def definition_arrays(self):
        # Create goals vectors
        self.array_states = tf.TensorArray(dtype=tf.float32, size=(FLAGS.time_horizon + 1), clear_after_read=False)
        self.array_utterances = tf.TensorArray(dtype=tf.float32, size=(FLAGS.time_horizon + 1), clear_after_read=False)
        self.array_mem_com = tf.TensorArray(dtype=tf.float32, size=(FLAGS.time_horizon + 1), clear_after_read=False)
        self.array_mem_last = tf.TensorArray(dtype=tf.float32, size=(FLAGS.time_horizon + 1), clear_after_read=False)

    def get_placeholders(self):
        [self.states, self.utterances, self.mem_com, self.mem_last, self.goal_types, self.goal_locations,
         self.full_goals, self.name_targets] = self.policy.get_placeholders()

    def write_arrays(self):
        self.array_states = self.array_states.write(0, self.states)
        self.array_utterances = self.array_utterances.write(0, self.utterances)
        self.array_mem_com = self.array_mem_com.write(0, self.mem_com)
        self.array_mem_last = self.array_mem_last.write(0, self.mem_last)

    def loop(self):
        t = tf.constant(0)
        return_sofar = tf.zeros([FLAGS.batch_size, 1], tf.float32)
        args = [self.array_states, self.array_utterances, self.array_mem_com, self.array_mem_last, self.goal_types,
                self.goal_locations, self.full_goals, self.name_targets, t, return_sofar]

        (array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations, full_goals,
         name_targets, t_fin, rewards_batch) = tf.while_loop(self.condition, self.body, args)

        reward = tf.reduce_mean(rewards_batch, axis=0)
        self.step = self.optimizer.minimize(-reward, global_step=self.global_step)
        self.array_states_stack = array_states.stack(),
        self.array_utterances_stack = array_utterances.stack(),
        self.array_mem_com_stack = array_mem_com.stack(),
        self.array_mem_last_stack = array_mem_last.stack(),
        self.t_fin = t_fin,
        self.reward = reward

        # return step, array_states.stack(), array_utterances.stack(), array_mem_com.stack(), array_mem_last.stack(), t, reward

    def body(self, array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations,
             full_goals,
             name_targets, t, return_sofar):

        # Reading the last state of environment
        states = array_states.read(t)
        utterances = array_utterances.read(t)
        mem_com = array_mem_com.read(t)
        mem_last = array_mem_last.read(t)

        # new_states = states
        # new_utterances = utterances
        # new_mem_com = mem_com
        # new_mem_last = mem_last

        # new_positions = tf.slice(new_states, [0, 0, 0], [FLAGS.number_agents + FLAGS.number_landmarks, FLAGS.dim_env, FLAGS.batch_size])
        # new_velocities = tf.slice(new_states, [0, 2, 0], [FLAGS.number_agents + FLAGS.number_landmarks, FLAGS.dim_env, FLAGS.batch_size])
        # new_gazes = tf.slice(new_states, [0, 4, 0], [FLAGS.number_agents + FLAGS.number_landmarks, FLAGS.dim_env, FLAGS.batch_size])
        # phys_output = tf.zeros([1, 4, FLAGS.batch_size])
        # new_states = tf.zeros(shape = [FLAGS.number_agents + FLAGS.number_landmarks, 9, FLAGS.batch_size])
        # new_utterances = tf.zeros(shape = [FLAGS.number_agents, 20, FLAGS.batch_size])
        # new_mem_com = tf.zeros(shape = [FLAGS.number_agents, 32, FLAGS.batch_size])
        # new_mem_last = tf.zeros(shape = [FLAGS.number_agents, 32, FLAGS.batch_size])

        phys_output, new_velocities, new_gazes, new_utterances, delta_mem_com, delta_mem_last = self.policy.forward_pass(
            states,
            utterances, mem_com, mem_last, full_goals)
        new_states, new_positions = compute_new_states(states, new_velocities, new_gazes, new_utterances)
        new_mem_com, new_mem_last = compute_new_memories(mem_com, mem_last, delta_mem_com, delta_mem_last)
        return_sofar += compute_reward(new_positions, new_gazes, phys_output, new_utterances, name_targets,
                                       goal_locations,
                                       goal_types)

        # Writing the new state
        array_states = array_states.write((t + 1), new_states)
        array_utterances = array_utterances.write((t + 1), new_utterances)
        array_mem_com = array_mem_com.write((t + 1), new_mem_com)
        array_mem_last = array_mem_last.write((t + 1), new_mem_last)

        t += 1

        return [array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations, full_goals,
                name_targets, t, return_sofar]

    def condition(self, array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations,
                  full_goals,
                  name_targets, t, return_sofar):
        return tf.less(t, FLAGS.time_horizon)

    def create_feed_dict(self, states, utterances, memories_com, memories_last, goal_locations, goal_types, targets):
        list_values = [states, utterances, memories_com, memories_last, goal_types, goal_locations, targets]
        list_placeholders = [self.states, self.utterances, self.mem_com, self.mem_last, self.goal_types,
                             self.goal_locations, self.name_targets]
        feed_dict = {a: b for a, b in zip(list_placeholders, list_values)}
        return feed_dict

    def train(self, sess):
        print("Initializing variables")
        sess.run(self.init)
        sess.graph.finalize()
        print("Start training")
        start = datetime.now()
        for i in range(FLAGS.max_steps):
            states, utterances, memories_com, memories_last, goal_locations, goal_types, targets = self.env.random_generation()
            generation_time = datetime.now() - start
            feed_dict = self.create_feed_dict(states, utterances, memories_com, memories_last, goal_locations,
                                              goal_types, targets)
            _, array_states, array_utterances, array_mem_com, array_mem_last, t, reward = sess.run(self.output_to_run,
                                                                                                   feed_dict)
            self.reward_history.append(reward)
            # self.env_history.append([states, utterances, memories_com, memories_last, goal_locations, goal_types, targets])
            # self.arrays_history.append([array_states, array_utterances, array_mem_com, array_mem_last])

            if i % FLAGS.print_frequency == 0:
                print("\n")
                print("iteration " + str(i))
                print(reward)
                states = array_states[0][-1, :, :, :]
                print_stats_agent(states, goal_locations, goal_types)
                print("computing time")
                print(datetime.now() - start)
                print("generation time")
                print(generation_time)
                print("memory usage")
                memory()

                start = datetime.now()