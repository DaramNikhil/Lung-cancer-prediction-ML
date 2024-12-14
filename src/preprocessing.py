import numpy as np
from sklearn.preprocessing import LabelEncoder

def data_preprocessing(data):
    data.columns = data.columns.str.lower()
    data.columns = data.columns.str.replace(" ", "_")
    data.rename(columns={"fatigue_": "fatigue", "allergy_": "allergy"}, inplace=True)
    
    label_encoder = LabelEncoder()
    data['gender'] = label_encoder.fit_transform(data['gender'])  
    data['lung_cancer'] = label_encoder.fit_transform(data['lung_cancer'])
    return data
