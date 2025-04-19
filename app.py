import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

class InferenceHandler:
    #Function untuk declare variabel yang akan digunakan
    def __init__(self, model_path, encoding_info, training_columns):
        self.model_path = model_path
        self.model = self.load_model()
        self.encoding_info = encoding_info
        self.training_columns = training_columns
        self.one_hot_encoders = {}

    #Digunakan untuk load model yang telah di save dalam bentuk pkl
    def load_model(self):
        with open(self.model_path, 'rb') as file:
            model = joblib.load(file)
        return model

    #Digunakan untuk encoding dari data yang sudah diinput
    def preprocess_data(self, new_data):
        label_encoder = LabelEncoder()
        for column in self.encoding_info['label_encode']:
            new_data[column] = label_encoder.fit_transform(new_data[column])

        enc_columns_list = []
        for column in self.encoding_info['one_hot_encode']:
            if column not in self.one_hot_encoders:
                enc = OneHotEncoder()
                enc.fit(new_data[[column]])
                self.one_hot_encoders[column] = enc

            enc = self.one_hot_encoders[column]
            encoded_df = pd.DataFrame(enc.transform(new_data[[column]]).toarray(), columns=enc.get_feature_names_out([column]))
            enc_columns_list.append(encoded_df)

        new_data = new_data.reset_index(drop=True)
        new_data = pd.concat([new_data] + enc_columns_list, axis=1)

        for column in self.encoding_info['one_hot_encode']:
            new_data = new_data.drop([column], axis=1)

        if 'index' in new_data.columns:
            new_data = new_data.drop('index', axis=1)

        new_data = new_data.reindex(columns=self.training_columns, fill_value=0)
        return new_data

    #Melakukan predict dari data yang telah dilakukan encoding
    def predict(self, new_data):
        processed_data = self.preprocess_data(new_data)
        predictions = self.model.predict(processed_data)
        return predictions

#Function untuk design dari website yang dibuat
def main():
    st.title('Wilbert Suwanto - 2702369756')
    st.title('Hotel Booking Prediction')

    no_of_adults = st.number_input('Number of Adults', min_value=0, max_value=10)
    no_of_children = st.number_input('Number of Children', min_value=0,  max_value=10)
    no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, max_value=7)
    no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, max_value=20)
    type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox('Required Car Parking Space', [0, 1])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
    lead_time = st.number_input('Lead Time', min_value=0, max_value=1000)
    arrival_year = st.number_input('Arrival Year', min_value=2000, max_value=2023, value=2022)
    arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=3)
    arrival_date = st.number_input('Arrival Date', min_value=1, max_value=31, value=28)
    market_segment_type = st.selectbox('Market Segment Type', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'])
    repeated_guest = st.number_input('Repeated Guest', min_value=0, max_value=1, value=0)
    no_of_previous_cancellations = st.number_input('Number of Previous Cancellations', min_value=0, max_value=20)
    no_of_previous_bookings_not_canceled = st.number_input('Number of Previous Bookings Not Canceled', min_value=0, max_value=100)
    avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, max_value=100000.00)
    no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, value=5)

    if st.button('Make Prediction'):
        new_data = pd.DataFrame({
            'no_of_adults': [no_of_adults],
            'no_of_children': [no_of_children],
            'no_of_weekend_nights': [no_of_weekend_nights],
            'no_of_week_nights': [no_of_week_nights],
            'type_of_meal_plan': [type_of_meal_plan],
            'required_car_parking_space': [required_car_parking_space],
            'room_type_reserved': [room_type_reserved],
            'lead_time': [lead_time],
            'arrival_year': [arrival_year],
            'arrival_month': [arrival_month],
            'arrival_date': [arrival_date],
            'market_segment_type': [market_segment_type],
            'repeated_guest': [repeated_guest],
            'no_of_previous_cancellations': [no_of_previous_cancellations],
            'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
            'avg_price_per_room': [avg_price_per_room],
            'no_of_special_requests': [no_of_special_requests]
        })

        model_path = 'RF_class.pkl'
        encoding_info = {
            'label_encode': {
                'arrival_year': LabelEncoder()
            },
            'one_hot_encode': ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        }
        training_columns = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'required_car_parking_space', 'lead_time',
                            'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                            'avg_price_per_room', 'no_of_special_requests', 'type_of_meal_plan_Meal Plan 1', 'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Meal Plan 3',
                            'type_of_meal_plan_Not Selected', 'room_type_reserved_Room_Type 1', 'room_type_reserved_Room_Type 2', 'room_type_reserved_Room_Type 3',
                            'room_type_reserved_Room_Type 4', 'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6', 'room_type_reserved_Room_Type 7',
                            'market_segment_type_Aviation', 'market_segment_type_Complementary', 'market_segment_type_Corporate', 'market_segment_type_Offline', 'market_segment_type_Online']
        
        inference_handler = InferenceHandler(model_path, encoding_info, training_columns)
        predictions = inference_handler.predict(new_data)

        prediction_label = "Not Canceled" if predictions[0] == 1 else "Canceled"
        st.success(f'The prediction is: {prediction_label}')

if __name__ == "__main__":
    main()