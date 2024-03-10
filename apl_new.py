import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
app = Flask(__name__)


@app.route('/submit-form', methods=['POST'])
def handle_data():
    data_dict = {}
    features=['Time', 'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
       'Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
       'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle',
       'Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
       'Road_allignment', 'Types_of_Junction', 'Road_surface_type',
       'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
       'Type_of_collision',  'Vehicle_movement', 'Casualty_class',
       'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity',
       'Work_of_casuality', 'Fitness_of_casuality', 'Pedestrian_movement',
       'Cause_of_accident', 'Accident_severity']
    adapt_data=np.load('adapt_data.npy')
    layer = tf.keras.layers.Normalization(axis=-1)
    layer.adapt(adapt_data)
    data=np.array([])
    # 从表单请求中提取数据
    form_data = request.get_json()
    # 将表单数据转换为字典
    data_dict = {key: form_data[key] for key in form_data}
    # 处理字典数据（例如打印或保存）
    print("Data received:", data_dict)
    label_encoders = {}

    for ft in features:
        label_encoders[ft] = joblib.load(f'enc/{ft}.pkl')
        
    for k in features:
        data_dict[k] = label_encoders[k].transform(data_dict[k])[0]
        
    values=[]
    
    for v in data_dict.values():
        values.append(v)
        
    normed_data=layer(np.array([values]))   
    
    for i in range(normed_data.shape[1]):
        data_dict[features[i]]=normed_data[0][i]
    
    model = tf.keras.models.load_model('acp')

    predict_value=model.predict(data_dict)

    return jsonify({'prediction': predict_value})
    
    
    
if __name__ == '__main__':
    app.run(debug=True)




