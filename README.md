# 🌾 SmartField AI: AI-Driven Precision Agriculture System 🌿

**SmartField AI** is a web-based platform that empowers farmers by integrating **machine learning** and **computer vision** to recommend optimal crops, predict crop yield, detect plant diseases, and suggest nutrient deficiencies and fertilizer use. This AI-powered tool helps farmers make informed decisions based on environmental and soil data.

![SmartField AI Banner](https://github.com/your-username/Smartfield-AI/assets/your-image-id) <!-- Optional: replace with screenshot -->

---

## 🚀 Features

- 🌱 **Crop Recommendation** based on NPK values, pH, rainfall, humidity, and temperature
- 🧪 **Fertilizer Suggestion** based on nutrient deficiency analysis
- 🍃 **Plant Disease Detection** using CNN and ResNet
- 🩺 **Nutrient Deficiency Detection** using VGG & custom CNN
- 📈 **Yield Prediction** using linear regression
- 🌦️ **Live Weather Integration** via OpenWeather API
- 🖼️ Image-based UI for user-friendly interaction

---

## 🛠 Technologies Used

| Category        | Tools/Tech                                          |
|----------------|-----------------------------------------------------|
| Web Framework  | Flask, HTML, CSS, Bootstrap, Jinja2                |
| ML Models      | Scikit-learn (Random Forest, Linear Regression)     |
| DL Models      | PyTorch (ResNet), Keras/TensorFlow (VGG)           |
| APIs           | OpenWeather API                                     |
| Image Handling | Pillow, Torchvision                                 |
| Data Handling  | Pandas, NumPy                                       |
| Model Storage  | `.pkl`, `.pth`, `.hdf5` model files                 |

---

## 📁 Project Structure
```
Smartfield-AI/
│
├── app.py # Flask main application
├── model.py # Crop recommendation model logic
├── fertilizer.py # Fertilizer logic
├── /models/ # Trained ML/DL model files (.pth, .hdf5, .pkl)
├── /templates/ # HTML pages (index.html, results.html, etc.)
├── /static/ # CSS, JS, Images
├── /data/ # CSV datasets (crop, fertilizer, yield)
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
---

## 💡 Modules Explained

### 🌾 1. Crop Recommendation
- **Model**: Random Forest Classifier
- **Input**: N, P, K, pH, rainfall, temperature, humidity
- **Data**: `Crop_recommendation.csv`
- **Output**: Best crop to grow
- **Live data**: Temperature & humidity fetched via OpenWeather API

### 🧪 2. Fertilizer Recommendation
- **Logic**: Rule-based comparison of actual vs. ideal NPK values
- **Output**: Suggests nutrient correction & matching fertilizer
- **Files**: `fertilizer.py`, `Fertilizer.csv`, `FertilizerData.csv`

### 🍃 3. Plant Disease Detection
- **Model**: CNN with ResNet
- **Input**: Leaf image (upload or webcam)
- **Output**: Disease name + cure
- **Model File**: `plant_disease_model.pth`

### 🌱 4. Nutrient Deficiency Detection
- **Model**: VGG CNN & Custom CNN
- **Input**: Deficient leaf image
- **Output**: Nutrient lacking (N/P/K)
- **Accuracy**: ~92% F2 Score
- **Model File**: `nutrition.hdf5`

### 📊 5. Yield Prediction
- **Model**: Linear Regression (also tested Random Forest Regressor)
- **Input**: Weather, soil data
- **Output**: Predicted crop yield (kg/hectare)
- **Data**: `yield_data.csv`

---
## 🏆 Achievements & Recognition

- ✅ **Accepted at GreenAI Nexus 2025**, SRM University – a national-level innovation forum promoting sustainable AI solutions in agriculture.
- ✅ **Selected for presentation at ITAI 2025** – *International Conference on Intelligent Technologies and Applications in Industry* (January 24–25, 2025)  
  Hosted by Gurugram University, Gurgaon, India  
  📖 Proceedings published in **SCOPUS-indexed Springer LNNS Series**  
  [Conference Link »](https://scrs.in/conference/itai2025)

## 🤝 Acknowledgments

- [OpenWeather API](https://openweathermap.org/) – For real-time temperature and humidity data
- [Kaggle Datasets](https://www.kaggle.com/) – For crop, fertilizer, and leaf disease image datasets
- [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [scikit-learn](https://scikit-learn.org/) – For building and training machine learning and deep learning models
- [PlantVillage Dataset](https://www.plantvillage.org/) – For disease classification and image training

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/softpratham/Smartfield-AI.git
cd Smartfield-AI

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py


