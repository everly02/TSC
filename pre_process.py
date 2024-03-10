import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

dataframe = pd.read_csv("RTA Dataset.csv")

#缺失值处理
for column in dataframe.columns:
    
    # 将空字符串替换为 unknown
    dataframe[column].replace('', 'unknown', inplace=True)
    dataframe[column].replace(pd.NA, 'unknown', inplace=True)
    dataframe[column].replace('Unknown', 'unknown', inplace=True)
#删除过多缺失的数据    
unknown_count = dataframe.apply(lambda row: row.eq('unknown').sum(), axis=1)

# 删除超过5个 'unknown' 的行
df_filtered = dataframe[unknown_count <= 5]

dataframe.reset_index(drop=True, inplace=True)

dataframe['Time'] = pd.to_datetime(dataframe['Time'])

dataframe['Time'] = dataframe['Time'].dt.hour

features = ['Time', 'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
       'Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
       'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle',
       'Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
       'Road_allignment', 'Types_of_Junction', 'Road_surface_type',
       'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
       'Type_of_collision',  'Vehicle_movement', 'Casualty_class',
       'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity',
       'Work_of_casuality', 'Fitness_of_casuality', 'Pedestrian_movement',
       'Cause_of_accident', 'Accident_severity']
# 编码器

label_encoders = {}

label_encoder = LabelEncoder()

# 对DataFrame中的每个列进行编码
for f in features:
    label_encoders[f] = LabelEncoder()
    dataframe[f] = label_encoders[f].fit_transform(dataframe[f])
    joblib.dump(label_encoders[f], f'enc/{f}.pkl')

#tf normalization
lb_arr=np.array(dataframe['Accident_severity'])
lb_arr=np.reshape(lb_arr, (-1, 1))

adapt_data=np.array(dataframe.drop('Accident_severity', axis=1))
np.save('adapt_data', adapt_data)
input_data=adapt_data
layer = tf.keras.layers.Normalization(axis=-1)
layer.adapt(adapt_data)

tmp=layer(input_data)
tmp=np.hstack([tmp,lb_arr])

dataframe=pd.DataFrame(tmp, columns=dataframe.columns)

dataframe.to_csv('new_data.csv', index=False)


