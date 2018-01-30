from trainer.src.util import activation_function
import tensorflow as tf


class PhysicalNet:
    def __init__(self):
        self.input_size = 3 * tf.app.flags.FLAGS.dim_env + tf.app.flags.FLAGS.color_size

        self.Weights = []
        self.Biases = []

        self.init_weights()
        self.init_biases()

    def init_weights(self):
        # Initialization of the weights of all the networks' layers
        # Weights are 3 dimensional arrays: [number of agents, number of units, number of inputs]
        # This shape enables us to handle all the agents/landmarks states at once, instead of dealing with list of agents' states
        with tf.variable_scope("phys_variable") as scope:
            for i in range(tf.app.flags.FLAGS.number_layers):
                if i == 0:
                    W = tf.tile(tf.get_variable("weight_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, self.input_size],
                                                initializer=tf.orthogonal_initializer()),
                                [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, 1, 1])
                    tf.summary.histogram('phys_net_weight_' + str(i), W)
                elif i != (tf.app.flags.FLAGS.number_layers - 1):
                    W = tf.tile(tf.get_variable("weight_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, tf.app.flags.FLAGS.layer_sizes],
                                                initializer=tf.orthogonal_initializer()),
                                [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, 1, 1])
                    tf.summary.histogram('phys_net_weight_' + str(i), W)
                else:
                    W = tf.tile(tf.get_variable("weight_" + str(i), shape=[1, tf.app.flags.FLAGS.output_size, tf.app.flags.FLAGS.layer_sizes],
                                                initializer=tf.orthogonal_initializer()),
                                [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, 1, 1])
                    tf.summary.histogram('phys_net_weight_' + str(i), W)

                self.Weights.append(W)

    def init_biases(self):
        # Initialization of the weights of all the networks' biases.
        # Same remark as the weights concerning the shapes of the biases.
        with tf.variable_scope("phys_variable") as scope:
            for i in range(tf.app.flags.FLAGS.number_layers):
                if i < (tf.app.flags.FLAGS.number_layers - 1):
                    B = tf.tile(tf.get_variable("bias_" + str(i), shape=[1, tf.app.flags.FLAGS.layer_sizes, 1],
                                                initializer=tf.orthogonal_initializer()),
                                [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, 1, 1])
                    tf.summary.histogram('phys_net_bias_' + str(i), B)
                else:
                    B = tf.tile(tf.get_variable("bias_" + str(i), shape=[1, tf.app.flags.FLAGS.output_size, 1],
                                                initializer=tf.orthogonal_initializer()),
                                [tf.app.flags.FLAGS.number_agents + tf.app.flags.FLAGS.number_landmarks, 1, 1])

                self.Biases.append(B)

    def compute_output(self, x):
        # Compute a forward pass through the network
        # Input: a tensor of shape [number of agents, size of input, batch _size]
        # Output: a tensor of shape [number of agents, output_size, batch_size]
        for i in range(tf.app.flags.FLAGS.number_layers):
            W = self.Weights[i]
            b = self.Biases[i]
            if i != (tf.app.flags.FLAGS.number_layers - 1):
                x = tf.nn.dropout(activation_function(tf.matmul(W, x) + b), keep_prob=tf.app.flags.FLAGS.keep_prob)
            else:
                x = activation_function(tf.matmul(W, x) + b)

        return x