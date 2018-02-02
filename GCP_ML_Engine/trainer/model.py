## Contains the TensorFlow graph code—the logic of the model

# Imports
from datetime import datetime
from trainer.src.environment import Environment
from trainer.src.policy import Policy
import tensorflow as tf
import numpy as np
import cPickle
from trainer.src.util import print_stats_agent, compute_new_states, compute_new_memories, compute_reward, memory, \
    delete_history_files, print_stat_vocabulary


class Experiment:
    def __init__(self):
        self.policy = Policy()
        self.env = Environment()
        delete_history_files()

        self.get_placeholders()
        self.definition_arrays()
        self.write_arrays()
        self.learning_rate = self.learning_rate_decay()
        tf.summary.scalar('learning rate', self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.loop()
        self.output_to_run = [self.step, self.array_states_stack, self.array_utterances_stack, self.array_mem_com_stack,
                              self.array_mem_last_stack,
                              self.f_g, self.array_outputs_stack, self.t_fin, self.reward]
        self.merged = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()

        self.reward_history = []
        self.env_history = []
        self.arrays_history = []

    def learning_rate_decay(self):
        self.global_step = tf.Variable(0, trainable=False)
        if tf.app.flags.FLAGS.learning_rate_decay:
            starter_learning_rate = tf.app.flags.FLAGS.learning_rate
            boundaries = [1000, 10000]
            values = [tf.app.flags.FLAGS.learning_rate, tf.app.flags.FLAGS.learning_rate / 10, tf.app.flags.FLAGS.learning_rate / 100]
            return tf.train.piecewise_constant(self.global_step, boundaries, values, name=None)
        else:
            return tf.app.flags.FLAGS.learning_rate

    def definition_arrays(self):
        # Create goals vectors
        self.array_states = tf.TensorArray(dtype=tf.float32, size=(tf.app.flags.FLAGS.time_horizon + 1), clear_after_read=False)
        self.array_utterances = tf.TensorArray(dtype=tf.float32, size=(tf.app.flags.FLAGS.time_horizon + 1), clear_after_read=False)
        self.array_mem_com = tf.TensorArray(dtype=tf.float32, size=(tf.app.flags.FLAGS.time_horizon + 1), clear_after_read=False)
        self.array_mem_last = tf.TensorArray(dtype=tf.float32, size=(tf.app.flags.FLAGS.time_horizon + 1), clear_after_read=False)
        self.array_outputs = tf.TensorArray(dtype=tf.float32, size=(tf.app.flags.FLAGS.time_horizon + 1), clear_after_read=False)

    def get_placeholders(self):
        [self.states, self.utterances, self.mem_com, self.mem_last, self.goal_types, self.goal_locations,
         self.full_goals, self.name_targets] = self.policy.get_placeholders()

    def write_arrays(self):
        self.array_states = self.array_states.write(0, self.states)
        self.array_utterances = self.array_utterances.write(0, self.utterances)
        self.array_mem_com = self.array_mem_com.write(0, self.mem_com)
        self.array_mem_last = self.array_mem_last.write(0, self.mem_last)
        self.array_outputs = self.array_outputs.write(0, np.zeros((tf.app.flags.FLAGS.number_agents, 4, tf.app.flags.FLAGS.batch_size),
                                                                  dtype=np.float32))

    def loop(self):
        t = tf.constant(0)
        return_sofar = tf.zeros([tf.app.flags.FLAGS.batch_size, 1], tf.float32)
        args = [self.array_states, self.array_utterances, self.array_mem_com, self.array_mem_last, self.goal_types,
                self.goal_locations, self.full_goals, self.name_targets, self.array_outputs, t, return_sofar]

        (array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations, full_goals,
         name_targets, array_outputs, t_fin, rewards_batch) = tf.while_loop(self.condition, self.body, args,
                                                                            parallel_iterations=1)


        reward = tf.reshape(tf.reduce_mean(rewards_batch, axis=0), [])
        tf.summary.scalar('accuracy', -reward)
        grads = self.optimizer.compute_gradients(-reward)
        self.step = self.optimizer.apply_gradients(grads, global_step=self.global_step)
        for index, grad in enumerate(grads):
            tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

        self.array_states_stack = array_states.stack()
        self.array_utterances_stack = array_utterances.stack()
        self.array_mem_com_stack = array_mem_com.stack()
        self.array_mem_last_stack = array_mem_last.stack()
        self.array_outputs_stack = array_outputs.stack()

        # voc_reward = dirichlet_log_lik(self.array_utterances_stack)
        self.f_g = full_goals
        self.t_fin = t_fin,
        self.reward = reward  # + voc_reward



    def body(self, array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations,
             full_goals,
             name_targets, array_outputs, t, return_sofar):

        # Reading the last state of environment
        states = array_states.read(t)
        utterances = array_utterances.read(t)
        mem_com = array_mem_com.read(t)
        mem_last = array_mem_last.read(t)

        phys_output, new_velocities, new_delta_gazes, new_utterances, delta_mem_com, delta_mem_last = self.policy.forward_pass(
            states,
            utterances, mem_com, mem_last, full_goals)

        new_states, new_positions, new_gazes = compute_new_states(states, new_velocities, new_delta_gazes,
                                                                  new_utterances)

        new_mem_com, new_mem_last = compute_new_memories(mem_com, mem_last, delta_mem_com, delta_mem_last)

        return_sofar += compute_reward(new_positions, new_gazes, phys_output, new_utterances, name_targets,
                                       goal_locations,
                                       goal_types)

        # Writing the new state
        array_states = array_states.write((t + 1), new_states)
        array_utterances = array_utterances.write((t + 1), new_utterances)
        array_mem_com = array_mem_com.write((t + 1), new_mem_com)
        array_mem_last = array_mem_last.write((t + 1), new_mem_last)
        array_outputs = array_outputs.write((t + 1), phys_output)

        t += 1

        return [array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations, full_goals,
                name_targets, array_outputs, t, return_sofar]

    def condition(self, array_states, array_utterances, array_mem_com, array_mem_last, goal_types, goal_locations,
                  full_goals,
                  name_targets, array_outputs, t, return_sofar):
        return tf.less(t, tf.app.flags.FLAGS.time_horizon)

    def create_feed_dict(self, states, utterances, memories_com, memories_last, goal_locations, goal_types, targets):
        list_values = [states, utterances, memories_com, memories_last, goal_types, goal_locations, targets]
        list_placeholders = [self.states, self.utterances, self.mem_com, self.mem_last, self.goal_types,
                             self.goal_locations, self.name_targets]
        feed_dict = {a: b for a, b in zip(list_placeholders, list_values)}
        return feed_dict

    def train(self, sess):
        self.train_writer = tf.summary.FileWriter('Summary', sess.graph)
        print("Initializing variables")
        sess.run(self.init)
        sess.graph.finalize()
        print("Start training")
        start = datetime.now()
        self.arrays_history = [0, 0, 0, 0, 0]
        self.full_g = []
        # states, utterances, memories_com, memories_last, goal_locations, goal_types, targets = self.env.random_generation()
        for i in range(tf.app.flags.FLAGS.max_steps):
            states, utterances, memories_com, memories_last, goal_locations, goal_types, targets = self.env.random_generation()
            generation_time = datetime.now() - start
            feed_dict = self.create_feed_dict(states, utterances, memories_com, memories_last, goal_locations,
                                              goal_types, targets)

            if i % tf.app.flags.FLAGS.tensorboard_freq == 0:
                # Here is the problem
                _, array_states, array_utterances, array_mem_com, array_mem_last, full_goals, array_outputs, t, reward, summary = sess.run(self.output_to_run + [self.merged], feed_dict)
                self.train_writer.add_summary(summary, i)
            else:
                _, array_states, array_utterances, array_mem_com, array_mem_last, full_goals, array_outputs, t, reward = sess.run(
                    self.output_to_run, feed_dict)
            self.reward_history.append(reward)
            self.full_g.append(self.arrays_history[-1])
            self.arrays_history = [array_states, array_utterances, array_mem_com, array_mem_last, full_goals,
                                   array_outputs]

            with open('env_history.pkl', 'wb') as f:
                pickler = cPickle.Pickler(f)
                pickler.dump([states, utterances, memories_com, memories_last, goal_locations, goal_types, targets])

            with open("arrays_history.pkl", 'wb') as f:
                pickler = cPickle.Pickler(f)
                pickler.dump([array_states, array_utterances, array_mem_com, array_mem_last, array_outputs])

            if i % tf.app.flags.FLAGS.print_frequency == 0:
                print("\n")
                print("iteration " + str(i))
                print(reward)
                final_states = array_states[-1, :, :, :]
                print_stats_agent(final_states, goal_locations, goal_types, targets)
                print_stat_vocabulary(array_utterances)
                print("computing time")
                print(datetime.now() - start)
                print("generation time")
                print(generation_time)
                print("memory usage")
                memory()

                start = datetime.now()