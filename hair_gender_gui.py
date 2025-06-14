import numpy as np
import cv2 as cv2
import cvlib as cv
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# === Load Models ===
age_gender_model = load_model("Age_gender.keras")
hair_classifier = load_model("hairDetect.keras")

# === Face Detection & Prediction Logic ===
def detect_face_and_predict(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    faces, _ = cv.detect_face(img_resized)

    if not faces:
        messagebox.showinfo("No Face", "No face detected in the image.")
        return None, None, None, img_resized

    x1, y1, x2, y2 = faces[0]
    face_crop = img_resized[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (110, 110))
    face_normalized = face_crop.astype(np.float32) / 255.0

    gender_pred, age_pred = age_gender_model.predict(np.expand_dims(face_normalized, axis=0))
    gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
    age = int(round(age_pred[0][0]))
    final_gender = gender
    box_color = "green"

    if 20 <= age <= 30:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (110, 110)).astype(np.float32) / 255.0
        hair_pred = hair_classifier.predict(np.expand_dims(gray_face, axis=0))
        has_long_hair = hair_pred[0][0] > 0.5
        final_gender = "Female" if has_long_hair else "Male"
        box_color = "red"

    return final_gender, age, box_color, img_resized

# === GUI Setup ===
root = Tk()
root.title("Hair & Gender Classifier")
root.geometry("650x700")
root.configure(bg="#1e272e")

img_panel = Label(root, bg="#dff9fb")
img_panel.pack(pady=20)

info_label = Label(root, text="", font=("Arial", 18), bg="#1e272e", fg="white")
info_label.pack(pady=10)

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    gender, age, color, img = detect_face_and_predict(file_path)
    if gender is None:
        return

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((300, 300))
    tk_img = ImageTk.PhotoImage(pil_img)
    img_panel.config(image=tk_img)
    img_panel.image = tk_img

    info_text = f"Gender: {gender}    Age: {age}"
    info_label.config(text=info_text, fg=color)

btn_load = Button(root, text="Upload Image", font=("Arial", 14), command=load_image, bg="#0be881", fg="black")
btn_load.pack(pady=10)

btn_exit = Button(root, text="Exit", font=("Arial", 14), command=root.quit, bg="#ff3f34", fg="white")
btn_exit.pack(pady=10)

root.mainloop()
