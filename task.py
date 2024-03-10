import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from keras import layers
from keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split

ds=pd.read_csv("new_data.csv")

layer=tf.keras.layers.CategoryEncoding(
          num_tokens=3, output_mode="one_hot")
labels_tensor = tf.constant(ds['Accident_severity'])
encoded_labels = layer(labels_tensor)

encoded_df = pd.DataFrame(encoded_labels.numpy())

# 替换原始 DataFrame 中的标签列
ds = pd.concat([ds, encoded_df], axis=1)
ds.drop(columns=['Accident_severity'], inplace=True)

X_ds=ds.drop(columns=[0, 1, 2])
y_ds=ds[[0, 1, 2]]
X_ds.to_numpy()
y_ds.to_numpy()
X_train, X_test = train_test_split(X_ds, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y_ds, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(80, activation='relu', input_shape=(31,)),
    tf.keras.layers.Dense(112, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=1000)

model.save('acp')

# 评估模型
model.evaluate(X_test, y_test)
