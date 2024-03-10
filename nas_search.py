import keras_tuner
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten())
  for i in range(1, hp.Int('num_layers', 1, 3)):
    model.add(keras.layers.Dense(
        units=hp.Int('units_' + str(i), 16, 256, 16),
        activation=hp.Choice('activation', ['relu', 'tanh'])
    )) 
  model.add(keras.layers.Dense(3, activation='softmax'))
  learning_rate = hp.Float("lr", min_value=0.005, max_value=0.03, sampling="log")
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
      loss='mse',
      metrics=['accuracy'],
    )
  return model
build_model(keras_tuner.HyperParameters())
tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=150)

ds=pd.read_csv("new_data.csv")
# Split your data into features and target
X = ds.drop(columns=['Accident_severity'])
y = ds['Accident_severity']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_np = X_train.values
X_val_np = X_val.values
y_train_np = y_train.values
y_val_np = y_val.values

def write_list_to_file(list, filename):
  with open(filename, 'w') as f:
    f.writelines('\t'.join(str(item) for item in row) + '\n' for row in list)

y_train_np = keras.utils.to_categorical(y_train_np, 3)
y_val_np = keras.utils.to_categorical(y_val_np, 3)

tuner.search(X_train_np, y_train_np, epochs=10, validation_data=(X_val_np, y_val_np))

models = tuner.get_best_models(num_models=2)
best_model = models[0]

tuner.results_summary()
