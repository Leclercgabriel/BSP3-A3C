import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
class ActorCritic:
    def __init__(self, alpha=0.01, gamma=0.99, action_space=4, observation_space=8, a1=64, a2=32, c1=64, c2=32, l=1, Async="A", index="0"):
        super(ActorCritic, self).__init__()
        K.clear_session()
        self.index = index
        self.count=0
        self.countLimit=l
        self.Async=Async

        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
        self.action = None
        self.actionSpace = [i for i in range (0, self.action_space)]

        self.actor_checkpoint_file = os.path.join("logs/"+self.index+"/actor")
        self.critic_checkpoint_file = os.path.join("logs/"+self.index+"/critic")

        self.aD1_dims = a1
        self.aD2_dims = a2
        self.aOutput_dims = action_space

        self.cD1_dims = c1
        self.cD2_dims = c2
        self.cOutput_dims = 1

        self.stateInput = Input(shape=(observation_space))

        self.aD1 = Dense(self.aD1_dims, activation=tf.keras.layers.LeakyReLU(), dtype=tf.float64)(self.stateInput)
        self.aD2 = Dense(self.aD2_dims, activation=tf.keras.layers.LeakyReLU(), dtype=tf.float64)(self.aD1)
        self.aOutput = Dense(self.aOutput_dims, activation="softmax", dtype=tf.float64)(self.aD2)

        self.cD1 = Dense(self.cD1_dims, activation=tf.keras.layers.LeakyReLU(), dtype=tf.float64)(self.stateInput)
        self.cD2 = Dense(self.cD2_dims, activation=tf.keras.layers.LeakyReLU(), dtype=tf.float64)(self.cD1)
        self.cOutput = Dense(self.cOutput_dims, activation=None, dtype=tf.float64)(self.cD2)

        self.Actor = Model(inputs=self.stateInput, outputs=self.aOutput)
        self.Critic = Model(inputs=self.stateInput, outputs=self.cOutput)

        self.Actor.summary()
        self.Critic.summary()
        
        self.Actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.Critic.compile(optimizer=Adam(learning_rate=self.alpha))


    def chooseAction(self, observations):
        state = tf.convert_to_tensor([observations])
        proba = self.Actor(state)

        actionProba = tfp.distributions.Categorical(probs=proba)
        self.action = actionProba.sample().numpy()[0]

        return self.action, actionProba, proba

    def aSave(self):
        print("...saving actor...")
        self.Actor.save_weights(self.actor_checkpoint_file)

    def aLoad(self, logDate):
        print("...loading actor...")
        if (logDate != None):
            self.Actor.load_weights(os.path.join("logs_backup_"+logDate+"/"+self.index+"/actor"))
        else:
            self.Actor.load_weights(self.actor_checkpoint_file)
    
    def cSave(self):
        print("...saving critic...")
        self.Critic.save_weights(self.critic_checkpoint_file)

    def cLoad(self, logDate):
        print("...loading critic...")
        if (logDate != None):
            self.Critic.load_weights(os.path.join("logs_backup_"+logDate+"/"+self.index+"/critic"))
        else:
            self.Critic.load_weights(self.critic_checkpoint_file)

    def save(self):
        print("...saving...")
        self.aSave()
        self.cSave()

    def load(self, logDate):
        print("...loading...")
        self.aLoad(logDate)
        self.cLoad(logDate)

    def learn(self, state, reward, state1, done):
        state = tf.convert_to_tensor([state], dtype=tf.float64)
        state1 = tf.convert_to_tensor([state1], dtype=tf.float64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float64)
        
        with tf.GradientTape(persistent=True) as tape:
            
            actionProba = self.Actor(state)

            V_value = self.Critic(state)
            V_value1 = self.Critic(state1)
            V_value = tf.squeeze(V_value)
            V_value1 = tf.squeeze(V_value1)

            actionProba = tfp.distributions.Categorical(probs=actionProba)
            logProba = actionProba.log_prob(self.action)

            advantage = reward + self.gamma*V_value1*(1-int(done)) - V_value

            aLoss = -logProba * advantage
            cLoss = advantage ** 2

        aGradient = tape.gradient(aLoss, self.Actor.trainable_variables)
        cGradient = tape.gradient(cLoss, self.Critic.trainable_variables)
        
        self.count += 1
        if self.Async == "A":
            if self.count == self.countLimit:
                self.Actor.optimizer.apply_gradients(zip(aGradient, self.Actor.trainable_variables))
                self.count = 0
            self.Critic.optimizer.apply_gradients(zip(cGradient, self.Critic.trainable_variables))
        elif self.Async == "C":
            self.Actor.optimizer.apply_gradients(zip(aGradient, self.Actor.trainable_variables))
            if self.count == self.countLimit:
                self.Critic.optimizer.apply_gradients(zip(cGradient, self.Critic.trainable_variables))
                self.count = 0

        del tape
        del state
        del state1
        del reward
        del V_value
        del V_value1
        del aGradient
        del cGradient
