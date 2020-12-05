"""
Modified from https://github.com/vermouth1992/drl-portfolio-management/blob/master/src/model/ddpg
"""

import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd

from ddpg.replay_buffer import ReplayBuffer
from data.utils import normalize_state

import os
import sys
fileDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(fileDir)

def build_summaries():
    critic_loss=tf.Variable(0.)
    actor_loss=tf.Variable(0.)
    reward=tf.Variable(0.)
    ep_ave_max_q=tf.Variable(0.)

    tf.summary.scalar('Reward',reward)
    tf.summary.scalar('Ep_ave_max_q',ep_ave_max_q)
    tf.summary.scalar('Actor_loss', actor_loss)
    tf.summary.scalar('Critic_loss', critic_loss)

    summary_vars=[reward,ep_ave_max_q,actor_loss, critic_loss,]
    summary_ops=tf.summary.merge_all()
    return summary_ops,summary_vars

class DDPG():
    """
    the training algorithm
    """
    def __init__(self, env, sess, actor, critic, actor_noise, config, nn,
                 model_save_path, summary_path, train_info_path):

        self.config = config
        self.model_save_path = model_save_path
        self.summary_path = summary_path
        self.train_info_path = train_info_path

        self.sess = sess
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.nn = nn

        self.summary_ops, self.summary_vars = build_summaries()
        self.sess.run(tf.global_variables_initializer())

    def train(self):

        writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

        self.actor.update_target_network()
        self.critic.update_target_network()

        num_episode = self.config['episode']
        batch_size = self.config['batch_size']
        gamma = self.config['gamma']
        self.buffer = ReplayBuffer(self.config['buffer_size'], np.random.seed(self.config['seed']))
        
        reward_set = []
        q_value_set = []
        loss_set = []

        for i in range(num_episode):
            info = self.env.reset()
            obs1, time_window, done = self.env.provider.f_step()

            ep_reward = 0
            ep_ave_max_q = 0
            ep_loss = 0
            for j in range(self.config['steps']):

                action0_ = info["weight"]
                action0 = np.expand_dims(action0_, axis=0)
                state1 = np.expand_dims(obs1, axis=0)
                norm_state1 = normalize_state(state1)

                action = self.actor.predict(input_num = state1.shape[0],
                                            state = norm_state1,
                                            previous_action=action0) + self.actor_noise()
                # step forward
                reward, info, done = self.env.f_step(obs1, action[0])
                state2 = np.expand_dims(info["obs2"], axis=0)
                norm_state2 = normalize_state(state2)

                # add to buffer: the normalize the price
                self.buffer.add(norm_state1, action, reward, done, action0, norm_state2)

                obs1 = info["obs2"]
                ep_reward += reward

                #if self.buffer.size() >= batch_size:
                if True:
                    # batch update
                    s_batch, a_batch, r_batch, t_batch, a0_batch, s2_batch = self.buffer.sample_batch(batch_size)

                    # Calculate targets
                    input_num = s2_batch.shape[0]
                    target_action = self.actor.predict_target(input_num, s2_batch, a_batch)
                    target_q = self.critic.predict_target(input_num, s2_batch, a_batch, target_action)

                    y_i = []
                    for k in range(input_num):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value = np.reshape(y_i, (input_num, 1))
                    critic_loss,q_value = self.critic.train(input_num, s_batch, a0_batch, a_batch, predicted_q_value)

                    ep_ave_max_q += np.amax(q_value)
                    ep_loss += critic_loss

                    # Update the actor policy using the sampled gradient
                    a_out = self.actor.predict(input_num, s_batch, a0_batch)
                    grads = self.critic.action_gradients(input_num, s_batch, a0_batch, a_out)
                    self.actor.train(input_num, s_batch, a0_batch, grads[0])

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                if done or j == self.config['steps'] - 1:
                    summary_str = self.sess.run(self.summary_ops,
                                                feed_dict={self.summary_vars[0]: ep_reward,
                                                           self.summary_vars[1]: ep_ave_max_q / float(j)})

                    writer.add_summary(summary_str, i)
                    writer.flush()
                    
                    reward_set.append(ep_reward)
                    q_value_set.append((ep_ave_max_q / float(j)))
                    loss_set.append(ep_loss/float(j) * 100)
                    print("------------------------------------------------------------------------------------------------")
                    print('Episode: {:d}, Reward: {:.4f}, Qmax: {:.4f}, loss: {:.4f}'.format(i, ep_reward, (ep_ave_max_q / float(j)), ep_loss/float(j) * 100))
                    break

        self.save_model()
        
        train_info = pd.DataFrame({"reward": reward_set, "q_value": q_value_set, "loss": loss_set})
        train_info.to_csv(self.train_info_path)
        print('Finish.')
        
        return train_info

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)


