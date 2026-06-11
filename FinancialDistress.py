
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ================= MAIN WINDOW =================
main = tk.Tk()
main.title("Financial Distress Prediction System")
main.geometry("1600x900")
main.config(bg="#f4f6f8")

# ================= GLOBAL VARIABLES =================
df = None
X_train = X_test = y_train = y_test = None
scaler = StandardScaler()

models = {}
accuracies = {}
precisions = {}
recalls = {}
f1_scores = {}

FEATURES = [f'x{i}' for i in range(1, 31)]

# ================= HEADER =================
header = tk.Frame(main, bg="#2c3e50", height=80)
header.pack(fill="x")

tk.Label(
    header,
    text="Financial Distress Prediction using Machine Learning",
    bg="#2c3e50",
    fg="white",
    font=("Segoe UI", 20, "bold")
).pack(pady=20)

# ================= SIDEBAR =================
sidebar = tk.Frame(main, bg="#34495e", width=260)
sidebar.pack(side="left", fill="y")

def sidebar_button(text, command, y):
    tk.Button(
        sidebar,
        text=text,
        command=command,
        font=("Segoe UI", 11, "bold"),
        bg="#1abc9c",
        fg="black",
        activebackground="#16a085",
        relief="flat",
        height=2,
        width=24
    ).place(x=15, y=y)

# ================= OUTPUT AREA =================
content = tk.Frame(main, bg="#f4f6f8")
content.pack(fill="both", expand=True)

output = ScrolledText(
    content,
    font=("Consolas", 11),
    bg="white",
    fg="black",
    width=120,
    height=40
)
output.pack(padx=20, pady=20, fill="both", expand=True)

def log(msg):
    output.insert(tk.END, msg + "\n")
    output.see(tk.END)

# ================= FUNCTIONS =================

def uploadDataset():
    global df
    output.delete("1.0", tk.END)
    path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if path:
        df = pd.read_csv(path)
        log("Dataset Loaded Successfully")
        log(f"Shape: {df.shape}")

def preprocessDataset():
    global X_train, X_test, y_train, y_test

    if df is None:
        messagebox.showerror("Error", "Upload dataset first")
        return

    data = df.drop(['Company', 'Time'], axis=1, errors='ignore')
    data.fillna(data.mean(), inplace=True)

    data['Financial Distress'] = data['Financial Distress'].apply(
        lambda x: 1 if x <= -0.3 else 0
    )

    X = data[FEATURES]
    y = data['Financial Distress']

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log("Preprocessing Completed")
    log(f"Training Samples : {len(X_train)}")
    log(f"Testing Samples  : {len(X_test)}")
    log(f"Distressed (Train): {sum(y_train)}")
    log(f"Healthy (Train)   : {len(y_train)-sum(y_train)}")

def evaluate(name, preds):
    a = accuracy_score(y_test, preds) * 100
    p = precision_score(y_test, preds, zero_division=0) * 100
    r = recall_score(y_test, preds, zero_division=0) * 100
    f = f1_score(y_test, preds, zero_division=0) * 100

    accuracies[name] = a
    precisions[name] = p
    recalls[name] = r
    f1_scores[name] = f

    log("")
    log(f"{name} Evaluation")
    log(f"Accuracy  : {a:.2f}%")
    log(f"Precision : {p:.2f}%")
    log(f"Recall    : {r:.2f}%")
    log(f"F1-Score  : {f:.2f}%")

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Healthy", "Distressed"],
        yticklabels=["Healthy", "Distressed"]
    )
    plt.title(name + " Confusion Matrix")
    plt.show()

# ================= MODELS =================

def trainKNN():
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    models['KNN'] = model
    evaluate("KNN", preds)

def trainNB():
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    models['Naive Bayes'] = model
    evaluate("Naive Bayes", preds)

def trainSVM():
    model = SVC(kernel='rbf', probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    models['SVM'] = model
    evaluate("SVM", preds)

def trainRF():
    model = RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    models['Random Forest'] = model
    evaluate("Random Forest", preds)


def predictTest():
    if not models:
        messagebox.showerror("Error", "Train a model first")
        return

    best = max(f1_scores, key=f1_scores.get)
    model = models[best]

    path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    test_df = pd.read_csv(path)[FEATURES]
    test_scaled = scaler.transform(test_df)

    probs = model.predict_proba(test_scaled)

    log("")
    log(f"Prediction using Best Model: {best}")

    for i, p in enumerate(probs):
        prob = p[1] * 100
        label = "Financially Distressed" if prob >= 50 else "Financially Healthy"
        log(f"Record {i+1}: {label} (Distress Probability: {prob:.2f}%)")

def graphComparison():
    df_plot = pd.DataFrame({
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1-Score": f1_scores
    })
    df_plot.plot(kind="bar", figsize=(10,6))
    plt.ylabel("Percentage")
    plt.title("Model Performance Comparison")
    plt.show()

# ================= SIDEBAR BUTTONS =================
sidebar_button("Upload Dataset", uploadDataset, 40)
sidebar_button("Preprocess Dataset", preprocessDataset, 100)
sidebar_button("Train KNN", trainKNN, 160)
sidebar_button("Train Naive Bayes", trainNB, 220)
sidebar_button("Train SVM", trainSVM, 280)
sidebar_button("Train Random Forest", trainRF, 340)
sidebar_button("Predict Test Dataset", predictTest, 400)
sidebar_button("Graph Comparison", graphComparison, 460)

main.mainloop()


