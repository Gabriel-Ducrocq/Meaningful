## Contains the trainer logic that manages the job

# Imports
import tensorflow as tf
from trainer.model import Experiment

# Initialise the parameters flags

# Reset the graphs
tf.reset_default_graph()

# Run the training
exp = Experiment()
with tf.Session() as sess:
    exp.train(sess)