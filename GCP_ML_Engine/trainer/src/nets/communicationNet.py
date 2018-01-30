import tensorflow as tf

from trainer.src.util import activation_function


class CommunicationNet:
    def __init__(self):
        self.Weights = []
        self.Biases = []
        self.Weight_read_mem = tf.tile(
            tf.get_variable("com_memory_read_weight", shape=[1, tf.app.flags.FLAGS.output_size, tf.app.flags.FLAGS.mem_size],
                            initializer=tf.orthogonal_initializer()),
            [tf.app.flags.FLAGS.number_agents, 1, 1])

        self.init_weights()
        self.init_biases()
        self.def_delta_mem()

    def init_weights(self):
        # Initialization of the weights of all the networks' layers
        # Weights are 3 dimensional arrays: [number of agents, number of units, vocabulary size]
        # This shape enables us to handle all the agents utterances at once, instead of dealing with list of agents' states
        with tf.variable_scope("com_variable") as scope:
            for i in range(tf.app.flags.FLAGS.number_layers):
                if i == 0:
                    W = tf.tile(
                        tf.get_variable("com_net_weight_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, tf.app.flags.FLAGS.vocabulary_size],
                                        initializer=tf.orthogonal_initializer()),
                        [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('com_net_weight_' + str(i), W)
                elif i != (tf.app.flags.FLAGS.number_layers - 1):
                    W = tf.tile(
                        tf.get_variable("com_net_weight_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, tf.app.flags.FLAGS.layer_sizes],
                                        initializer=tf.orthogonal_initializer()),
                        [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('com_net_weight_' + str(i), W)
                else:
                    W = tf.tile(
                        tf.get_variable("com_net_weight_" + str(i), shape=[1, tf.app.flags.FLAGS.output_size, tf.app.flags.FLAGS.layer_sizes],
                                        initializer=tf.orthogonal_initializer()),
                        [tf.app.flags.FLAGS.number_agents, 1, 1])

                    tf.summary.histogram('com_net_weight_' + str(i), W)

                self.Weights.append(W)

    def init_biases(self):
        # Initialization of the weights of all the networks' biases.
        # Same remark as the weights concerning the shapes of the biases.
        with tf.variable_scope("com_variable") as scope:
            for i in range(tf.app.flags.FLAGS.number_layers):
                if i < (tf.app.flags.FLAGS.number_layers - 1):
                    B = tf.tile(tf.get_variable("com_net_bias_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, 1],
                                                initializer=tf.orthogonal_initializer()),
                                [tf.app.flags.FLAGS.number_agents, 1, 1])
                    tf.summary.histogram('com_net_bias_' + str(i), B)
                else:
                    B = tf.tile(tf.get_variable("com_net_bias_" + str(i), shape=[1, tf.app.flags.FLAGS.output_size, 1],
                                                initializer=tf.orthogonal_initializer()),
                                [tf.app.flags.FLAGS.number_agents, 1, 1])

                self.Biases.append(B)

    def def_delta_mem(self):
        # Initialization of the weights and biases writing in the memory.
        # Their shape are of the form [number of agents, memory_size, output size] and [number of agents, output size, 1]
        # So that we can handle the memories of all agents at onces instead of dealing with list of memories.
        self.W_mem = tf.tile(tf.get_variable("weight_mem_com", shape=[1, tf.app.flags.FLAGS.mem_size, tf.app.flags.FLAGS.output_size],
                                             initializer=tf.orthogonal_initializer()),
                             [tf.app.flags.FLAGS.number_agents, 1, 1])
        self.b_mem = tf.tile(tf.get_variable("bias_mem_com", shape=[1, tf.app.flags.FLAGS.mem_size, 1],
                                             initializer=tf.orthogonal_initializer()),
                             [tf.app.flags.FLAGS.number_agents, 1, 1])

    def compute_output(self, x, memory):
        for i in range(tf.app.flags.FLAGS.number_layers):
            W = self.Weights[i]
            b = self.Biases[i]
            if i != (tf.app.flags.FLAGS.number_layers - 1):
                x = tf.nn.dropout(activation_function(tf.matmul(W, x) + b), keep_prob=tf.app.flags.FLAGS.keep_prob)
            else:
                x = activation_function(tf.matmul(W, x) + tf.matmul(self.Weight_read_mem, memory) + b)

        delta_mem = activation_function(tf.add(tf.matmul(self.W_mem, x), self.b_mem))
        return x, delta_mem