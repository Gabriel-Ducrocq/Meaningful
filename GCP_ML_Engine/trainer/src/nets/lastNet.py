from trainer.src.util import activation_function
import tensorflow as tf


class LastNet:
    def __init__(self):
        self.input_size = 2 * tf.app.flags.FLAGS.output_size + tf.app.flags.FLAGS.color_size + tf.app.flags.FLAGS.number_goal_types + tf.app.flags.FLAGS.dim_env
        self.output_size = 2 * tf.app.flags.FLAGS.dim_env + tf.app.flags.FLAGS.vocabulary_size
        self.Weights = []
        self.Biases = []
        self.Weight_read_mem = tf.tile(
            tf.get_variable("reading_last_mem_weight", shape=[1, self.output_size, tf.app.flags.FLAGS.last_mem_size],
                            initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
            [tf.app.flags.FLAGS.number_agents, 1, 1])

        self.init_weights()
        self.init_biases()
        self.def_delta_mem()

    def init_weights(self):
        with tf.variable_scope("last_variable") as scope:
            for i in range(tf.app.flags.FLAGS.number_layers):
                if i == 0:
                    W = tf.tile(
                        tf.get_variable("last_net_weight_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, self.input_size],
                                        initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
                        [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_weight_' + str(i), W)
                elif i != (tf.app.flags.FLAGS.number_layers - 1):
                    W = tf.tile(
                        tf.get_variable("last_net_weight_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, tf.app.flags.FLAGS.layer_sizes],
                                        initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
                        [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_weight_' + str(i), W)
                else:
                    W = tf.tile(
                        tf.get_variable("last_net_weight_" + str(i), shape=[1, self.output_size, tf.app.flags.FLAGS.layer_sizes],
                                        initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
                        [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_weight_' + str(i), W)

                self.Weights.append(W)

    def init_biases(self):
        with tf.variable_scope("last_variable") as scope:
            for i in range(tf.app.flags.FLAGS.number_layers):
                if i != (tf.app.flags.FLAGS.number_layers - 1):
                    B = tf.tile(tf.get_variable("last_net_bias_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, 1],
                                                initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
                                [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_bias_' + str(i), B)
                else:
                    B = tf.tile(tf.get_variable("last_net_bias_" + str(i), shape=[1, self.output_size, 1],
                                                initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
                                [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('last_net_bias_' + str(i), B)

                self.Biases.append(B)

    def def_delta_mem(self):
        self.W_mem = tf.tile(tf.get_variable("weight_mem_last", shape=[1, tf.app.flags.FLAGS.last_mem_size, self.output_size],
                                             initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
                             [tf.app.flags.FLAGS.number_agents, 1, 1])
        self.b_mem = tf.tile(tf.get_variable("bias_mem_last", shape=[1, tf.app.flags.FLAGS.last_mem_size, 1],
                                             initializer=tf.contrib.layers.xavier_initializer(tf.app.flags.FLAGS.xav_init)),
                             [tf.app.flags.FLAGS.number_agents, 1, 1])

    def compute_output(self, x, memory):
        for i in range(tf.app.flags.FLAGS.number_layers):
            W = self.Weights[i]
            b = self.Biases[i]
            if i != (tf.app.flags.FLAGS.number_layers - 1):
                x = tf.nn.dropout(activation_function(tf.matmul(W, x) + b), keep_prob=tf.app.flags.FLAGS.keep_prob)
            else:
                x = activation_function(tf.matmul(W, x) + tf.matmul(self.Weight_read_mem, memory) + b)

        delta_mem = tf.add(tf.matmul(self.W_mem, x), self.b_mem)
        return x, delta_mem