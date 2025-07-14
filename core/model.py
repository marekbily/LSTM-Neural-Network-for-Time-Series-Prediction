import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import Huber
from tensorflow import keras as keras
import matplotlib.pyplot as plt

opt_map = {
    'adam': Adam,
    'sgd': SGD,
    'rmsprop': RMSprop
}

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

			optimizer_cfg = configs['model'].get('optimizer', {})
			learning_rate = optimizer_cfg.get('learning_rate', 0.001)
			optimizer = Adam(learning_rate=learning_rate)	
			opt_type = optimizer_cfg.get('type', 'adam').lower()
			opt_class = opt_map.get(opt_type, Adam)
			optimizer = opt_class(learning_rate=learning_rate)
			loss = configs['model']['loss']
			if loss == "huber_loss":
				loss_fn = Huber()
			else:
				loss_fn = loss  # e.g. "mse", "mae", etc.

		self.model.compile(
			loss=loss_fn,
			optimizer=optimizer,
			metrics=["mae", "mse", keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()]
		)


		print('[Model] Model Compiled')
		timer.stop()

	#dd
	'''def train(self, x, y, epochs, batch_size, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		history = self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)

		plt.plot(history.history["mae"], label="MAE")
		plt.plot(history.history["mse"], label="MSE")
		plt.plot(history.history["loss"], label="Loss")
		plt.xlabel("Epochs")
		plt.ylabel("Metric Value")
		plt.legend()
		plt.title("Model Convergence Over Epochs")
		plt.show()

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()'''
	
	#TODO: revamp similarly to train_generator
	'''
	def train(self, x, y, epochs, batch_size, save_dir, validation_split=0.1):
			timer = Timer()
			timer.start()
			print('[Model] Training Started')
			print(f'[Model] {epochs} epochs, {batch_size} batch size, validation_split={validation_split}')

			save_fname = os.path.join(
				save_dir,
				f"{dt.datetime.now().strftime('%d%m%Y-%H%M%S')}-e{epochs}.h5"
			)

			callbacks = [
				EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
				ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
			]

			history = self.model.fit(
				x, y,
				epochs=epochs,
				batch_size=batch_size,
				validation_split=validation_split,
				callbacks=callbacks
    			validation_data=val_data_gen,
    			validation_steps=val_steps,
    			steps_per_epoch=steps_per_epoch,
			)

			# Plot training & validation metrics
			epochs_range = range(1, len(history.history['loss']) + 1)
			plt.figure(figsize=(12, 4))

			# Loss
			plt.subplot(1, 3, 1)
			plt.plot(epochs_range, history.history['loss'], label='train_loss')
			plt.plot(epochs_range, history.history['val_loss'], label='val_loss')
			plt.title('Loss')
			plt.xlabel('Epochs')
			plt.ylabel('Loss')
			plt.legend()

			# MAE
			plt.subplot(1, 3, 2)
			plt.plot(epochs_range, history.history['mae'], label='train_mae')
			plt.plot(epochs_range, history.history['val_mae'], label='val_mae')
			plt.title('Mean Absolute Error')
			plt.xlabel('Epochs')
			plt.legend()

			# RMSE
			plt.subplot(1, 3, 3)
			plt.plot(epochs_range, history.history['rmse'], label='train_rmse')
			plt.plot(epochs_range, history.history['val_rmse'], label='val_rmse')
			plt.title('Root Mean Squared Error')
			plt.xlabel('Epochs')
			plt.legend()

			plt.tight_layout()
			plt.show()

			print(f'[Model] Training Completed. Model saved as {save_fname}')
			timer.stop()
	'''

	def train_generator(
		self,
		data_gen,
		epochs,
		batch_size,
		steps_per_epoch,
		save_dir,
		validation_data=None,
		validation_steps=None
	):
		timer = Timer()
		timer.start()
		print('[Model] Generator Training Started')
		print(f'[Model] {epochs} epochs, {batch_size} batch size, '
			f'{steps_per_epoch} steps/epoch, validation_steps={validation_steps}')

		# 1) Build filename + callbacks
		save_fname = os.path.join(
			save_dir,
			f"{dt.datetime.now():%d%m%Y-%H%M%S}-e{epochs}.h5"
		)
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]

		# 2) Fit – **must** pass validation_data & validation_steps to get val_loss
		history = self.model.fit(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			validation_data=validation_data,
			validation_steps=validation_steps,
			callbacks=callbacks
		)

		# 3) Plot only metrics that exist
		hist = history.history
		epochs_range = range(1, len(hist['loss']) + 1)

		plt.figure(figsize=(10, 6))
		# Training vs Validation Loss
		plt.plot(epochs_range, hist['loss'],  label='train_loss')
		if 'val_loss' in hist:
			plt.plot(epochs_range, hist['val_loss'],  label='val_loss')

		# MAE
		plt.plot(epochs_range, hist['mae'],  label='train_mae')
		if 'val_mae' in hist:
			plt.plot(epochs_range, hist['val_mae'],  label='val_mae')

		# RMSE (ensure you named it “rmse” in compile)
		plt.plot(epochs_range, hist['rmse'],  label='train_rmse')
		if 'val_rmse' in hist:
			plt.plot(epochs_range, hist['val_rmse'],  label='val_rmse')

		plt.xlabel('Epochs')
		plt.ylabel('Metric Value')
		plt.legend()
		plt.title('Generator Training Metrics')
		plt.tight_layout()
		plt.show()

		print(f'[Model] Generator Training Completed. Model saved as {save_fname}')
		timer.stop()
