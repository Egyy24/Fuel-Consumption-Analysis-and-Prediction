import gradio as gr
import joblib
import pandas as pd

filename = 'fuel_consumption_prediction_model.pkl'
model_components = joblib.load(filename)

model = model_components['model']
preprocessor = model_components['preprocessor']
feature_names_after_encoding = model_components['feature_names_after_encoding']

print("Model dan preprocessor berhasil dimuat.")

# Fungsi prediksi
def predict_fuel_consumption(engine_size, cylinders, vehicle_class, transmission, fuel_type):
    input_df = pd.DataFrame({
        'ENGINESIZE': [engine_size],
        'CYLINDERS': [cylinders],
        'VEHICLECLASS': [vehicle_class],
        'TRANSMISSION': [transmission],
        'FUELTYPE': [fuel_type]
    })

    input_processed = preprocessor.transform(input_df)

    if hasattr(input_processed, "toarray"):
        input_processed = input_processed.toarray()

    input_processed_df = pd.DataFrame(input_processed, columns=feature_names_after_encoding)

    # Melakukan prediksi
    prediction = model.predict(input_processed_df)[0]
    return f"{prediction:.2f} L/100km"


# Opsi dropdown (sesuai kategori hasil training, termasuk 'Other')
vehicle_class_options = [
    'COMPACT', 'FULL-SIZE', 'MID-SIZE', 'MINICOMPACT', 'MINIVAN', 'Other',
    'PICKUP TRUCK - SMALL', 'PICKUP TRUCK - STANDARD', 'STATION WAGON - SMALL',
    'SUBCOMPACT', 'SUV - SMALL', 'SUV - STANDARD', 'TWO-SEATER',
    'VAN - CARGO', 'VAN - PASSENGER'
]

transmission_options = [
    'A4', 'A5', 'A6', 'A7', 'A8', 'AM7', 'AS6', 'AS7', 'AS8',
    'AV', 'AV6', 'M5', 'M6', 'Other'
]

# FUELTYPE: X=Regular gasoline, Z=Premium gasoline, D=Diesel, E=Ethanol (E85)
fuel_type_options = ['X', 'Z', 'D', 'E']

demo = gr.Interface(
    fn=predict_fuel_consumption,
    inputs=[
        gr.Slider(minimum=1.0, maximum=8.4, step=0.1, value=2.0, label="Ukuran Mesin / Engine Size (L)"),
        gr.Slider(minimum=3, maximum=12, step=1, value=4, label="Jumlah Silinder / Cylinders"),
        gr.Dropdown(choices=vehicle_class_options, value='COMPACT', label="Kelas Kendaraan / Vehicle Class"),
        gr.Dropdown(choices=transmission_options, value='A6', label="Jenis Transmisi"),
        gr.Dropdown(choices=fuel_type_options, value='X', label="Jenis Bahan Bakar (X=Reguler, Z=Premium, D=Diesel, E=Ethanol)")
    ],
    outputs=gr.Textbox(label="Prediksi Konsumsi BBM Gabungan"),
    title="Prediksi Konsumsi Bahan Bakar Kendaraan",
    description="Masukkan spesifikasi kendaraan untuk memprediksi konsumsi BBM gabungan (kota + tol) dalam L/100km menggunakan model Random Forest Regressor.",
    examples=[
        [2.0, 4, 'COMPACT', 'A6', 'X'],
        [3.5, 6, 'SUV - STANDARD', 'AS6', 'Z'],
        [5.7, 8, 'PICKUP TRUCK - STANDARD', 'A6', 'X'],
    ]
)

demo.launch(debug=True)