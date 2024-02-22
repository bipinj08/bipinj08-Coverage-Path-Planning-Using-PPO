import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, Conv3D
import numpy as np
from src.CPP.State import CPPState
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import copy


class PPOAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5

        # Global-Local Map
        self.use_global_local = True
        self.global_map_scaling = 3
        self.local_map_size = 17

class PPOAgent(object):
    def __init__(self, params, example_state: CPPState, example_action, stats=None):
        self.params = params
        gamma = tf.constant(0.99, dtype=float)

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.scalars = example_state.get_num_scalars()
        #self.num_actions = len(type(example_action))
        self.num_map_channels = self.boolean_map_shape[2]
        #self.scalars = example_state.get_num_scalars()
        self.action_size = len(type(example_action))

        # Create shared inputs
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        states = [boolean_map_input,
                  scalars_input]

        # Initialization
        self.state_size = states
        self.max_average = 0  # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.shuffle = False
        self.optimizer = Adam

        # Create Actor-Critic network models
        self.Actor = Actor(input_states=self.state_size,
                           num_actions=self.action_size,
                           params=self.params,
                           optimizer=self.optimizer)
        self.Critic = Critic(input_states=self.state_size,
                             num_actions=self.action_size,
                             params=self.params,
                             optimizer=self.optimizer)

        if stats:
            stats.set_model(self.Actor.model)


    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        if type(state) == CPPState:
            state = self.Actor.transfrom_state(state, for_prediction=True)
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        #print(prediction, 'is predictionnnnnnnn')
        return action, action_onehot, prediction


    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def  replay(self, states, actions, rewards, predictions, dones, next_states):
        # Get Critic network predictions
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        self.Actor.model.fit(states, y_true, epochs=10, verbose=0, shuffle=self.shuffle)
        self.Critic.model.fit([states, values, target],
                               y=None,
                               epochs=10,
                               verbose=0,
                               shuffle=self.shuffle)


    def train(self, replay_memory):
        states = [np.asarray([state_oi.get_boolean_map() for state_oi in replay_memory[0]]).astype('float32'),
                  np.asarray([state_oi.movement_budget for state_oi in replay_memory[0]]).astype('float32')]
        actions = replay_memory[1]
        rewards = replay_memory[2]
        next_states = [np.asarray([state_oi.get_boolean_map() for state_oi in replay_memory[3]]).astype('float32'),
                  np.asarray([state_oi.movement_budget for state_oi in replay_memory[3]]).astype('float32')]
        dones = [state_oi.is_terminal() for state_oi in replay_memory[3]]
        predictions = replay_memory[4]
        self.replay(states, actions, rewards, predictions, dones, next_states)

    def save_weights(self, path_to_weights):
        self.Critic.model.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.Critic.model.save(path_to_model)


class PartModel:
    def __init__(self, params):
        self.params = params

    def create_map_proc(self, conv_in, name):
        global_map = tf.stop_gradient(AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

        self.global_map = global_map
        self.total_map = conv_in

        for k in range(self.params.conv_layers):
            global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                strides=(1, 1),
                                name=name + 'global_conv_' + str(k + 1))(global_map)

        flatten_global = Flatten(name=name + 'global_flatten')(global_map)

        crop_frac = float(self.params.local_map_size) / float(conv_in.shape[1])
        local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
        self.local_map = local_map

        for k in range(self.params.conv_layers):
            local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                               strides=(1, 1),
                               name=name + 'local_conv_' + str(k + 1))(local_map)

        flatten_local = Flatten(name=name + 'local_flatten')(local_map)

        return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])

    def build_model(self, bool_map, scalars, inputs, num_actions, name):
        flatten_map = self.create_map_proc(bool_map, name)
        layer = Concatenate(name=name + 'concat')([flatten_map, scalars])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(int(self.params.hidden_layer_size/(k+1)), activation='relu',
                          name=name + 'hidden_layer_all_' + str(k),
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01))(layer)
        if name == "actor":
            output = Dense(num_actions, activation="softmax")(layer)
            model = Model(inputs=inputs, outputs=output)
        else:
            old_values = Input(shape=(1,))
            y_true = Input(shape=(1,))
            output = Dense(1, activation=None)(layer)
            model = Model(inputs=[inputs, old_values, y_true], outputs=output)
            return output, y_true, old_values, model

        return model

    def transfrom_state(self, state: CPPState, for_prediction=False):
        bool_map = state.get_boolean_map()
        scalars = np.array(state.get_scalars(), dtype=np.single)
        state = [bool_map, scalars]
        if for_prediction:
            state = [state_oi[tf.newaxis, ...] for state_oi in state]
        return state


class Actor(PartModel):
    def __init__(self, input_states, num_actions, params, optimizer):
        super().__init__(params)
        self.action_space = num_actions
        self.model = self.build_model(bool_map=input_states[0],
                                      scalars=input_states[1],
                                      inputs=input_states,
                                      num_actions=num_actions,
                                      name="actor")
        self.model.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=self.params.learning_rate))

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:,1 + self.action_space:] #unpacking the y_true
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.model.predict(state)


class Critic(PartModel):
    def __init__(self, input_states, num_actions, params, optimizer):
        super().__init__(params)
        self.action_space = num_actions
        out, y_true, old_value, self.model = self.build_model(bool_map=input_states[0],
                                                              scalars=input_states[1],
                                                              inputs=input_states,
                                                              num_actions=num_actions,
                                                              name="critic")
        self.model.add_loss(self.__class__.critic_PPO2_loss(
            values=old_value,
            y_true=y_true,
            y_pred=out
        ))
        self.model.compile(loss=None,optimizer=optimizer(lr=self.params.learning_rate))

    @staticmethod
    def critic_PPO2_loss(y_true, y_pred, values):
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
        v_loss1 = (y_true - clipped_value_loss) ** 2
        v_loss2 = (y_true - y_pred) ** 2

        value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
        # value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def predict(self, state):
        final_model = Model([self.model.input[0], self.model.input[1]], self.model.output)
        return final_model.predict([state, np.zeros((state[0].shape[0], 1))])


