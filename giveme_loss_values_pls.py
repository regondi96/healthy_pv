#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


# In[2]:


# Import data from the path
missing_values = ["0"]
data = pd.read_csv("/Users/hogar/Desktop/datasetNREL.csv", na_values = missing_values)
data.head()


# In[3]:


# Set the datetime
data['Datetime'] = pd.to_datetime(data['Unnamed: 0'])
data = data.set_index('Datetime')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.head()


# In[4]:


data.drop(['Solar Zenith Angle'], axis=1, inplace=True)
data.drop(['Wind Direction'], axis=1, inplace=True)
data.drop(['Wind Speed'], axis=1, inplace=True)
data.drop(['Cloud Type'], axis=1, inplace=True)
data.head()


# In[6]:


df = data.dropna()
df.head()


# In[7]:


target_names = ['DNI', 'Clearsky DNI', 'Temperature']
shift_days = 2
shift_steps = shift_days * 22  # Number of steps.


# In[9]:


df_targets = df[target_names].shift(-shift_steps)


# In[10]:


df[target_names].head(shift_steps + 5)


# In[14]:


df_targets.tail()


# In[15]:


x_data = df.values[0:-shift_steps]
print(type(x_data))
print("Shape:", x_data.shape)


# In[16]:


y_data = df_targets.values[:-shift_steps]
print(type(y_data))
print("Shape:", y_data.shape)


# In[17]:


num_data = len(x_data)
num_data


# In[18]:


train_split = 0.9
num_train = int(train_split * num_data)
num_train


# In[19]:


num_test = num_data - num_train
num_test


# In[20]:


x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)


# In[21]:


y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)


# In[22]:


num_x_signals = x_data.shape[1]
num_x_signals


# In[23]:


num_y_signals = y_data.shape[1]
num_y_signals


# In[25]:


print("Min:", np.min(x_train))
print("Max:", np.max(x_train))


# In[26]:


x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))


# In[27]:


x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# In[28]:


print(x_train_scaled.shape)
print(y_train_scaled.shape)


# In[29]:


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)


# In[30]:


batch_size = 128
# No. of observations
sequence_length = 22 * 7 * 4
sequence_length


# In[31]:


generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)


# In[32]:


x_batch, y_batch = next(generator)


# In[33]:


print(x_batch.shape)
print(y_batch.shape)


# In[34]:


batch = 0   # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)


# In[35]:


seq = y_batch[batch, :, signal]
plt.plot(seq)


# In[36]:


validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


# In[37]:


model = Sequential()


# In[38]:


model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))


# In[39]:


model.add(Dense(num_y_signals, activation='sigmoid'))


# In[40]:


if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))


# In[41]:


warmup_steps = 50


# In[42]:


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# In[43]:


optimizer = RMSprop(lr=1e-3)


# In[44]:


model.compile(loss=loss_mse_warmup, optimizer=optimizer)


# In[45]:


model.summary()


# In[46]:


path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


# In[47]:


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)


# In[48]:


callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)


# In[49]:


callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)


# In[50]:


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# In[52]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(generator=generator,\n                    epochs=20,\n                    steps_per_epoch=100,\n                    validation_data=validation_data,\n                    callbacks=callbacks)')


# In[53]:


try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# In[54]:


result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))


# In[55]:


print("loss (test-set):", result)


# In[56]:


# If you have several metrics you can use this instead.
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))


# In[352]:


# Plot
def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(14,4))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='Real')
        plt.plot(signal_pred, label='Pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.05)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


# In[355]:


plot_comparison(start_idx=106014, length=52, train=True)


# In[390]:


plot_comparison(start_idx=100070, length=88, train=True)


# In[ ]:





# In[ ]:




