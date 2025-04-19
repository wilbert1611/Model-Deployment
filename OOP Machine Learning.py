import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import os
import joblib

class DataHandler:
    #Function untuk declare variabel yang akan digunakan
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    #Digunakan untuk read data pada dataset
    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    #Digunakan untuk drop missing data pada dataset
    def drop_missing_data(self):
        self.data = self.data.dropna()

    #Digunakan untuk drop column yang tidak digunakan
    def drop_column(self, column_name):
        column_name in self.data.columns
        self.data = self.data.drop(column_name, axis=1)
    
    #Digunakan untuk encoding pada label dari dataset
    def label_encode(self, column_name):
        label_encoder = preprocessing.LabelEncoder()
        self.data[column_name] = label_encoder.fit_transform(self.data[column_name])
    
    #Membedakan data input dan data output yang akan digunakan nanti ketika split data
    def create_input_output(self, target_column):
        self.input_df = self.data.drop(target_column, axis=1)
        self.output_df = self.data[target_column]

class ModelHandler:
    #Function untuk declare variabel yang akan digunakan
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    
    #Digunakan untuk split data antara data train dan data test
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_data, self.output_data, test_size=test_size, random_state=random_state)
    
    #Melakukan one hot encoding pada data kategori yang lebih dari 2 unique value
    def one_hot_encode(self, columns_to_encode):
        enc = preprocessing.OneHotEncoder()

        train_enc_list = []
        test_enc_list = []

        for column in columns_to_encode:
            train_enc = pd.DataFrame(enc.fit_transform(self.x_train[[column]]).toarray(), columns=enc.get_feature_names_out(input_features=[column]))
            test_enc = pd.DataFrame(enc.transform(self.x_test[[column]]).toarray(), columns=enc.get_feature_names_out(input_features=[column]))

            train_enc_list.append(train_enc)
            test_enc_list.append(test_enc)

        self.x_train = self.x_train.reset_index()
        self.x_test = self.x_test.reset_index()

        for train_enc, test_enc in zip(train_enc_list, test_enc_list):
            self.x_train = pd.concat([self.x_train, train_enc], axis=1)
            self.x_test = pd.concat([self.x_test, test_enc], axis=1)

        for column in columns_to_encode:
            self.x_train = self.x_train.drop(column, axis=1)
            self.x_test = self.x_test.drop(column, axis=1)

    #Melakukan label encode pada data yang hanya memiliki 2 unique value
    def label_encode(self, columns_to_encode):
        label_encoder = preprocessing.LabelEncoder()
        self.x_train[columns_to_encode] = label_encoder.fit_transform(self.x_train[columns_to_encode])
        self.x_test[columns_to_encode] = label_encoder.transform(self.x_test[columns_to_encode])

    #Digunakan untuk drop kolom index pada data, dikarenakan setelah dilakukan encoding, umumnya akan ada kolom index pada data tersebut
    def drop_index(self, column):
        self.x_train = self.x_train.drop(column, axis = 1)
        self.x_test = self.x_test.drop(column, axis = 1)
    
    #Digunakan untuk melakukan index ulang antara data train dan data test, hal tersebut dilakukan untuk mencegah adanya perbedaan jumlah
    #kolom antara data test dan data train
    def reindex(self):
        self.x_test = self.x_test.reindex(columns=self.x_train.columns, fill_value=0)

    #Digunakan untuk membuat model random forest
    def createModel(self):
         self.RF_class = RandomForestClassifier(random_state = 42)

    #Melakukan fit data untuk dilakukan training terhadap model yang telah dibuat
    def train_model(self):
        self.RF_class.fit(self.x_train, self.y_train)

    #Melakukan prediksi dari hasil training tadi menggunakan data test
    def makePrediction(self):
        self.y_predict = self.RF_class.predict(self.x_test) 

    #Mengevaluasi model berdasarkan besarnya accuracy
    def evaluate_model(self):
        pred = self.RF_class.predict(self.x_test)
        return accuracy_score(self.y_test, pred)
    
    #Membuat classification report dari hasil model yang telah dibuat
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['Canceled','Not_Canceled']))
    
    #Save model yang telah dibuat menggunakan joblib agar file bisa dikompresi
    def save_model_to_file(self, save_path, filename):
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'wb') as file:
            joblib.dump(self.RF_class, file)

#Memuat data dan model
file_path = 'C:/Users/Wilbert/UTS Model/Dataset_B_hotel.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.drop_missing_data()
data_handler.drop_column('Booking_ID')
data_handler.label_encode('booking_status')
data_handler.create_input_output('booking_status')

input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()
columns_to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
model_handler.one_hot_encode(columns_to_encode)
model_handler.label_encode('arrival_year')
model_handler.drop_index('index')
model_handler.reindex()
model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()
print("Model Accuracy:", model_handler.evaluate_model())
savepath = "C:/Users/Wilbert/UTS Model"
model_handler.save_model_to_file(savepath, 'RF_class.pkl')