from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = joblib.load("rf_model.pkl")

scaler_params = {
    "Edad": {"mean": 35.2, "std": 10.1},
    "Ingreso": {"mean": 750000, "std": 200000},
    "Antiguedad_Meses": {"mean": 48.0, "std": 25.0},
    "CantidadTelefonos": {"mean": 1.4, "std": 0.5},
    "CantidadHijos": {"mean": 2.1, "std": 1.0}
}

def estandarizar(valor, feature):
    mean = scaler_params[feature]["mean"]
    std = scaler_params[feature]["std"]
    return (valor - mean) / std

def transform_input_to_model_format(user_input):
    """
    Transform simplified user input to the normalized format expected by the model.
    
    Input format:
    {
        "Edad": 35,                     // Age in years
        "Genero": 1,                    // 1=Masculino, 2=Femenino  
        "EstadoCivil": 1,               // 1=Soltero(a), 2=Casado(a), 3=Divorciado(a), 4=Viudo(a), 5=Reconciliacion Judicial, 6=Separacion Judicial, 7=Otro
        "CantidadHijos": 0,             // Number of children
        "CantidadTelefonos": 1,         // Number of phones
        "Trabaja": true,                // Currently working (boolean)
        "Ingreso": 300000,              // Monthly income in colones
        "Antiguedad_Meses": 12,         // Work tenure in months
        "Trabajo_Fisico": false,        // Physical work (boolean)
        "Provincia": 1,                 // 1=San José, 2=Alajuela, 3=Cartago, 4=Heredia, 5=Guanacaste, 6=Puntarenas, 7=Limón
        "Patrono": 2,                   // 1=ATV, 2=Gobierno, 3=Independiente, 4=Privado
        "Hacienda_Inscrito": true,      // Registered with Hacienda (boolean)
        "nivel_ingreso": 3,             // Income level (1-5)
        "riesgo_despido": 2,            // Unemployment risk (1-5) 
        "movilidad_social": 4           // Social mobility (1-5)
    }
    """
    
    # Initialize all features with default values
    model_input = {
        "Edad": 0,
        "Trabaja": 0,
        "Ingreso": 0,
        "Antiguedad_Meses": 0,
        "Genero": 0,
        "Vive_GAM": 0,
        "CantidadTelefonos": 0,
        "CantidadHijos": 0,
        "Hacienda_Inscrito": 0,
        "Patrono_ATV": False,
        "Patrono_Gobierno": False,
        "Patrono_Independiente": False,
        "Patrono_Privado": False,
        "EstadoCivil_Casado(a)": False,
        "EstadoCivil_Divorciado(a)": False,
        "EstadoCivil_Reconciliacion Judicial": False,
        "EstadoCivil_Separacion Judicial": False,
        "EstadoCivil_Soltero(a)": False,
        "EstadoCivil_Viudez": False,
        "Provincia_ALAJUELA": False,
        "Provincia_CARTAGO": False,
        "Provincia_CONSULADO": False,
        "Provincia_GUANACASTE": False,
        "Provincia_HEREDIA": False,
        "Provincia_LIMON": False,
        "Provincia_PUNTARENAS": False,
        "Provincia_SAN JOSE": False,
        "es_profesional": 0,
        "nivel_ingreso": 0,
        "riesgo_despido": 0,
        "movilidad_social": 0,
        "trabajo_fisico": 0
    }
    
    # Age normalization (standardization - you may need to adjust these parameters based on your training data)
    edad = user_input.get('Edad', 35)
    model_input["Edad"] = estandarizar(edad, "Edad")
    
    # Gender mapping (1=Masculino, 2=Femenino)
    genero = user_input.get('Genero', 1)
    model_input["Genero"] = 1 if genero == 1 else 0  # 1 for Masculino, 0 for Femenino
    
    # Employment status
    trabaja = user_input.get('Trabaja', False)
    model_input["Trabaja"] = 1 if trabaja else 0
    
    # Income normalization (standardization)
    ingreso = user_input.get('Ingreso', 0)
    model_input["Ingreso"] = estandarizar(ingreso, "Ingreso")
    
    # Work tenure normalization
    antiguedad = user_input.get('Antiguedad_Meses', 0)
    model_input["Antiguedad_Meses"] = estandarizar(antiguedad, "Antiguedad_Meses")
    
    # Phone count normalization
    telefonos = user_input.get('CantidadTelefonos', 1)
    model_input["CantidadTelefonos"] = estandarizar(telefonos, "CantidadTelefonos")
    
    # Children count normalization
    hijos = user_input.get('CantidadHijos', 0)
    model_input["CantidadHijos"] = estandarizar(hijos, "CantidadHijos")
    
    # Hacienda registration
    hacienda = user_input.get('Hacienda_Inscrito', False)
    model_input["Hacienda_Inscrito"] = 1 if hacienda else 0
    
    # Physical work
    trabajo_fisico = user_input.get('Trabajo_Fisico', False)
    model_input["trabajo_fisico"] = 1 if trabajo_fisico else 0
    
    # Civil status one-hot encoding
    estado_civil = user_input.get('EstadoCivil', 1)
    if estado_civil == 1:  # Soltero(a)
        model_input["EstadoCivil_Soltero(a)"] = True
    elif estado_civil == 2:  # Casado(a)
        model_input["EstadoCivil_Casado(a)"] = True
    elif estado_civil == 3:  # Divorciado(a)
        model_input["EstadoCivil_Divorciado(a)"] = True
    elif estado_civil == 4:  # Viudo(a)
        model_input["EstadoCivil_Viudez"] = True
    elif estado_civil == 5:  # Reconciliacion Judicial
        model_input["EstadoCivil_Reconciliacion Judicial"] = True
    elif estado_civil == 6:  # Separacion Judicial
        model_input["EstadoCivil_Separacion Judicial"] = True
    # Note: "Otro" (7) doesn't have a specific column, so it remains all False
    
    # Province one-hot encoding and GAM area calculation
    provincia = user_input.get('Provincia', 1)
    gam_provinces = [1, 2, 3, 4]  # San José, Alajuela, Cartago, Heredia are GAM
    
    if provincia == 1:  # San José
        model_input["Provincia_SAN JOSE"] = True
        model_input["Vive_GAM"] = 1
    elif provincia == 2:  # Alajuela
        model_input["Provincia_ALAJUELA"] = True
        model_input["Vive_GAM"] = 1
    elif provincia == 3:  # Cartago
        model_input["Provincia_CARTAGO"] = True
        model_input["Vive_GAM"] = 1
    elif provincia == 4:  # Heredia
        model_input["Provincia_HEREDIA"] = True
        model_input["Vive_GAM"] = 1
    elif provincia == 5:  # Guanacaste
        model_input["Provincia_GUANACASTE"] = True
        model_input["Vive_GAM"] = 0
    elif provincia == 6:  # Puntarenas
        model_input["Provincia_PUNTARENAS"] = True
        model_input["Vive_GAM"] = 0
    elif provincia == 7:  # Limón
        model_input["Provincia_LIMON"] = True
        model_input["Vive_GAM"] = 0
    
    # Employer type one-hot encoding
    patrono = user_input.get('Patrono', 1)
    if patrono == 1:  # ATV
        model_input["Patrono_ATV"] = True
    elif patrono == 2:  # Gobierno
        model_input["Patrono_Gobierno"] = True
    elif patrono == 3:  # Independiente
        model_input["Patrono_Independiente"] = True
    elif patrono == 4:  # Privado
        model_input["Patrono_Privado"] = True
    
    # Risk factors (direct mapping)
    model_input["nivel_ingreso"] = user_input.get('nivel_ingreso', 1)
    model_input["riesgo_despido"] = user_input.get('riesgo_despido', 1)
    model_input["movilidad_social"] = user_input.get('movilidad_social', 1)
    
    # Professional status (could be derived from income level or other factors)
    # For now, setting to 0 as default
    model_input["es_profesional"] = 0
    
    return model_input

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict credit risk based on user input.
    
    Expected JSON body structure:
    {
        "Edad": 35,                     // Age in years (number)
        "Genero": 1,                    // Gender: 1=Masculino, 2=Femenino
        "EstadoCivil": 1,               // Marital status: 1=Soltero(a), 2=Casado(a), 3=Divorciado(a), 4=Viudo(a), 5=Reconciliacion Judicial, 6=Separacion Judicial, 7=Otro
        "CantidadHijos": 0,             // Number of children (number)
        "CantidadTelefonos": 1,         // Number of phones (number)
        "Trabaja": true,                // Currently working (boolean)
        "Ingreso": 300000,              // Monthly income in colones (number)
        "Antiguedad_Meses": 12,         // Work tenure in months (number)
        "Trabajo_Fisico": false,        // Physical work (boolean)
        "Provincia": 1,                 // Province: 1=San José, 2=Alajuela, 3=Cartago, 4=Heredia, 5=Guanacaste, 6=Puntarenas, 7=Limón
        "Patrono": 2,                   // Employer: 1=ATV, 2=Gobierno, 3=Independiente, 4=Privado
        "Hacienda_Inscrito": true,      // Registered with Hacienda (boolean)
        "nivel_ingreso": 3,             // Income level (1-5, from risk factors sliders)
        "riesgo_despido": 2,            // Unemployment risk (1-5, from risk factors sliders)
        "movilidad_social": 4           // Social mobility (1-5, from risk factors sliders)
    }
    
    Returns:
    {
        "mal_pagador": boolean,         // True if predicted as bad payer
        "probabilidad": float           // Probability of being a bad payer (0-1)
    }
    """
    try:
        user_input = request.json
        
        # Transform the simplified input to the complex model format
        model_input = transform_input_to_model_format(user_input)
        
        # Create DataFrame for prediction
        df = pd.DataFrame([model_input])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return jsonify({
            "mal_pagador": bool(prediction),
            "probabilidad": round(probability, 4)
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error processing prediction: {str(e)}"
        }), 400

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for Fly.io"""
    return jsonify({
        "status": "healthy",
        "service": "predictor-api",
        "version": "1.0.0"
    })

@app.route("/predict_raw", methods=["POST"])
def predict_raw():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])  # Espera un JSON con los mismos campos que usaste para entrenar
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return jsonify({
            "mal_pagador": bool(prediction),
            "probabilidad": round(probability, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
