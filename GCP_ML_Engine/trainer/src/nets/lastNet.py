class LastNet:
    def __init__(self):
        self.input_size = 2 * FLAGS.output_size + FLAGS.color_size + FLAGS.number_goal_types + FLAGS.dim_env
        self.output_size = 2 * FLAGS.dim_env + FLAGS.vocabulary_size
        self.Weights = []
        self.Biases = []
        self.Weight_read_mem = tf.tile(
            tf.get_variable("reading_last_mem_weight", shape=[1, self.output_size, FLAGS.last_mem_size],
                            initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
            [FLAGS.number_agents, 1, 1])

        self.init_weights()
        self.init_biases()
        self.def_delta_mem()

    def init_weights(self):
        with tf.variable_scope("last_variable") as scope:
            for i in range(FLAGS.number_layers):
                if i == 0:
                    W = tf.tile(
                        tf.get_variable("last_net_weight_" + str(i), shape=[1, FLAGS.layer_sizes, self.input_size],
                                        initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
                        [FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_weight_' + str(i), W)
                elif i != (FLAGS.number_layers - 1):
                    W = tf.tile(
                        tf.get_variable("last_net_weight_" + str(i), shape=[1, FLAGS.layer_sizes, FLAGS.layer_sizes],
                                        initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
                        [FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_weight_' + str(i), W)
                else:
                    W = tf.tile(
                        tf.get_variable("last_net_weight_" + str(i), shape=[1, self.output_size, FLAGS.layer_sizes],
                                        initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
                        [FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_weight_' + str(i), W)

                self.Weights.append(W)

    def init_biases(self):
        with tf.variable_scope("last_variable") as scope:
            for i in range(FLAGS.number_layers):
                if i != (FLAGS.number_layers - 1):
                    B = tf.tile(tf.get_variable("last_net_bias_" + str(i), shape=[1, FLAGS.layer_sizes, 1],
                                                initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
                                [FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_bias_' + str(i), B)
                else:
                    B = tf.tile(tf.get_variable("last_net_bias_" + str(i), shape=[1, self.output_size, 1],
                                                initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
                                [FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_bias_' + str(i), B)

                self.Biases.append(B)

    def def_delta_mem(self):
        self.W_mem = tf.tile(tf.get_variable("weight_mem_last", shape=[1, FLAGS.last_mem_size, self.output_size],
                                             initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
                             [FLAGS.number_agents, 1, 1])
        self.b_mem = tf.tile(tf.get_variable("bias_mem_last", shape=[1, FLAGS.last_mem_size, 1],
                                             initializer=tf.contrib.layers.xavier_initializer(FLAGS.xav_init)),
                             [FLAGS.number_agents, 1, 1])

    def compute_output(self, x, memory):
        for i in range(FLAGS.number_layers):
            W = self.Weights[i]
            b = self.Biases[i]
            if i != (FLAGS.number_layers - 1):
                x = tf.nn.dropout(activation_function(tf.matmul(W, x) + b), keep_prob=FLAGS.keep_prob)
            else:
                x = activation_function(tf.matmul(W, x) + tf.matmul(self.Weight_read_mem, memory) + b)

        delta_mem = tf.add(tf.matmul(self.W_mem, x), self.b_mem)
        return x, delta_mem