# **Predictor API**  

API de predicción de morosidad desarrollada en **Flask**, que utiliza un modelo **Random Forest** previamente entrenado y almacenado en formato `.pkl`. Este prototipo fue desplegado en **Fly.io** y está pensado para integrarse con una interfaz gráfica (frontend) y para pruebas con herramientas como Postman.

## 🚀 Descripción  

Este proyecto es un prototipo funcional para predecir la probabilidad de que un cliente incurra en morosidad, usando datos reales de una institución bancaria costarricense (no datos sintéticos).  

El modelo fue entrenado previamente y serializado como un archivo `.pkl` de **263 MB**, por lo que se gestiona a través de **GitHub LFS** debido a su tamaño.  

La API expone endpoints REST para enviar datos de un cliente y recibir como respuesta:  
- `mal_pagador` → valor booleano que indica si es probable que el cliente sea moroso.  
- `probabilidad` → probabilidad estimada de morosidad (0 a 1).  

---

## 🛠 Tecnologías utilizadas  

- **Python 3.11**  
- **Flask** (framework web)  
- **pandas / numpy** (procesamiento de datos)  
- **scikit-learn** (modelo Random Forest)  
- **joblib** (serialización del modelo)  
- **Flask-CORS** (manejo de CORS)  
- **GitHub LFS** (gestión de archivos grandes)  
- **Docker** (empaquetado y despliegue)  
- **Fly.io** (infraestructura y hosting)  

---

## 📂 Estructura del proyecto  

```
predictor-api/
│
├── app.py                  # Código principal de Flask
├── rf_model.pkl            # Modelo entrenado (gestionado con Git LFS)
├── requirements.txt        # Dependencias
├── Dockerfile              # Imagen para despliegue en Fly.io
├── fly.toml                # Configuración de Fly.io
└── README.md               # Este archivo
```

---

## ⚙️ Instalación y ejecución local  

### 1. Clonar el repositorio con soporte para Git LFS
```bash
git clone <URL_DEL_REPOSITORIO>
cd predictor-api
git lfs install
git lfs pull
```

### 2. Crear y activar un entorno virtual  
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
venv\Scripts\activate     # En Windows
```

### 3. Instalar dependencias  
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la API localmente  
```bash
python app.py
```
Por defecto se levanta en `http://0.0.0.0:5000`.

---

## 🧪 Pruebas con Postman  

Puedes probar la API enviando un `POST` a:  
```
http://localhost:5000/predict
```

**Ejemplo de cuerpo JSON**:
```json
{
  "Edad": 35,
  "Genero": 1,
  "EstadoCivil": 1,
  "CantidadHijos": 0,
  "CantidadTelefonos": 1,
  "Trabaja": true,
  "Ingreso": 300000,
  "Antiguedad_Meses": 12,
  "Trabajo_Fisico": false,
  "Provincia": 1,
  "Patrono": 2,
  "Hacienda_Inscrito": true,
  "nivel_ingreso": 3,
  "riesgo_despido": 2,
  "movilidad_social": 4
}
```

---

## ☁️ Despliegue en Fly.io  

1. Inicia sesión en Fly.io:
```bash
fly auth login
```

2. Crea la app (si no existe):
```bash
fly launch
```

3. Ajusta `fly.toml` para usar al menos **2048 MB de RAM**:
```bash
fly scale memory 2048
fly scale vm shared-cpu-1x
```

4. Despliega:
```bash
fly deploy
```

---

## ⚠️ Limitaciones del prototipo  

- No incluye autenticación ni control de acceso.  
- No implementa registro de actividad ni auditoría.  
- No está optimizado para modelos muy ligeros; el `.pkl` es pesado y puede generar **cold start** en ciertos entornos.  
- Pensado para **demostración académica**, no para producción bancaria real.

---

## 📜 Licencia  
Este proyecto es parte de un trabajo académico para la Maestría en Machine Learning en OBS. Su uso en producción debe evaluarse considerando la privacidad de datos, la regulación vigente y la seguridad de la información.
