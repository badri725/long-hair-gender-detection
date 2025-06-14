# Long Hair-Based Gender Classification (Age-Sensitive)

## 📌 Description

This project implements a custom-trained AI system to detect gender based on hair length and age. It uses two models:
- A CNN that predicts gender and age from face images
- A binary classifier that detects long vs. short hair

Special logic is applied:
- For people **aged 20 to 30**:
  - Gender is inferred from **hair length** (e.g., long-haired male is classified as female)
- For people **outside this range**:
  - Gender is predicted directly from the model

---

## 🧠 Model Details

- `Age_gender.keras`: Predicts gender (male/female) and approximate age
- `hairDetect.keras`: Predicts if hair is long or short (binary classifier)
- Face detection is done using `cvlib`

## Download Trained Models

Download the `.keras` models and place them in your project folder:

- [Age_gender.keras](https://drive.google.com/file/d/13u4O6f7GOmxQ5PxBoKw2k-VeQNjyX28V/view?usp=sharing)
- [hairDetect.keras](https://drive.google.com/file/d/1bZmWOMIjVubzJ0Ftmy3tcZ1tTb5YLl6m/view?usp=sharing)


---

## 🖼️ GUI Features

Built using **Tkinter**:
- Upload an image with a visible face
- Automatically detects the face, age, and hair
- Displays:
  - Original image with predictions
  - Gender and age in color-coded format

---

## 🧪 How to Use

### 🔹 1. Install Requirements

```bash
pip install -r requirements.txt
