## Contains the trainer logic that manages the job

# Imports
import tensorflow as tf
from trainer.model import Experiment

# Initialise the parameters flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 50000, 'Number of iteration to train.')
flags.DEFINE_integer('number_layers', 3, 'Number of layers in each network')
flags.DEFINE_integer('layer_sizes', 256, 'Number of units in hidden layer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size.')
flags.DEFINE_integer('dim_env', 2, 'dimension of the environment')
flags.DEFINE_integer('number_goal_types', 3, 'number of different goal types')
flags.DEFINE_integer('color_size', 3, 'number of components of the color: RGB as usual')
flags.DEFINE_integer("output_size", 256, "number of units in the output layer")
flags.DEFINE_float("keep_prob", 0.9, "Dropouts rate of keeping")
flags.DEFINE_boolean("xav_init", False,"Distribution of initialization: False for normal, True for uniform" )
flags.DEFINE_integer("number_agents", 1, "Number of agents in the environment")
flags.DEFINE_integer("number_landmarks", 1, "Number of landmarks in the environment")
flags.DEFINE_integer("vocabulary_size", 20, "Size of the vocabulary")
flags.DEFINE_integer("mem_size", 32, "Size of the communication network's memory")
flags.DEFINE_integer("last_mem_size", 32, "Size of the last network's memory")
flags.DEFINE_float("gumbel_temperature", 1, "Temperature use for the gumbel softmax trick")
flags.DEFINE_float("sddev_phys_sampling", 0.0001, "Standard deviation used to sample the velocity and gaze output")
flags.DEFINE_float("delta_t", 0.1, "delta of time between timesteps")
flags.DEFINE_float("damping_coef", 0.5, "damping coefficient for the new velocity computation")
flags.DEFINE_float("stddev_memory", 0.0001, "standard deviation of the gaussian used to update memories")
flags.DEFINE_integer("bound", 5, "Bounds of generation of initial positions, centered in 0.")
flags.DEFINE_integer("time_horizon", 50, "Number of timestep before the end of the experiment.")
flags.DEFINE_integer("print_frequency", 500, "Frequency at which we print the reward, in number of steps.")
flags.DEFINE_boolean("learning_rate_decay", True, "Wether to use a piecewise learning rate decay or no decay at all")

# Reset the graphs
tf.reset_default_graph()

exp = Experiment()
with tf.Session() as sess:
    exp.train(sess)