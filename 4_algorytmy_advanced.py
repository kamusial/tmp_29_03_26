import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)


# ============================================================
# USTAWIENIA PROGRAMU
# ============================================================

RANDOM_STATE = 0
TEST_SIZE = 0.2

print("=" * 80)
print("START PROGRAMU")
print("Program porówna 4 algorytmy klasyfikacji:")
print("1. Regresja logistyczna")
print("2. KNN")
print("3. Drzewo decyzyjne")
print("4. SVM")
print("=" * 80)


# ============================================================
# 1. PRZYGOTOWANIE DANYCH
# ============================================================

print("\nKROK 1: Przygotowanie danych")

# Przykładowe dane testowe.
# Studenci mogą ten fragment zamienić na wczytanie własnych danych z pliku CSV.
X, y = make_circles(
    n_samples=5000,
    factor=0.2,
    random_state=RANDOM_STATE,
    noise=0.6
)

print(f"Utworzono zbiór danych.")
print(f"Liczba obserwacji: {X.shape[0]}")
print(f"Liczba cech: {X.shape[1]}")
print(f"Liczba klas: {len(set(y))}")

# ------------------------------------------------------------
# Przykład użycia własnych danych:
#
# df = pd.read_csv("moje_dane.csv")
# X = df.drop("target", axis=1)
# y = df["target"]
#
# gdzie "target" to nazwa kolumny z klasą decyzyjną.
# ------------------------------------------------------------


# ============================================================
# 2. WIZUALIZACJA DANYCH
# ============================================================

print("\nKROK 2: Wizualizacja danych")

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Wizualizacja danych")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.show()

print("Wykres został wyświetlony.")


# ============================================================
# 3. PODZIAŁ DANYCH NA ZBIÓR TRENINGOWY I TESTOWY
# ============================================================

print("\nKROK 3: Podział danych na zbiór treningowy i testowy")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} obserwacji")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]} obserwacji")


# ============================================================
# 4. DEFINICJA ALGORYTMÓW
# ============================================================

print("\nKROK 4: Przygotowanie modeli")

# Pipeline ze StandardScaler jest używany tam, gdzie skala danych ma duże znaczenie.
# Dotyczy to szczególnie: regresji logistycznej, KNN i SVM.

models = {
    "Regresja logistyczna": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000
        )
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "Drzewo decyzyjne": DecisionTreeClassifier(
        max_depth=None,
        random_state=RANDOM_STATE
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )
    )
}

print("Modele zostały przygotowane.")
print("Studenci mogą zmieniać hiperparametry modeli, np.:")
print("- n_neighbors w KNN")
print("- max_depth w drzewie decyzyjnym")
print("- C, gamma i kernel w SVM")
print("- max_iter lub C w regresji logistycznej")


# ============================================================
# 5. FUNKCJA DO OCENY MODELU
# ============================================================

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Funkcja trenuje model, wykonuje predykcję i oblicza metryki jakości.

    Obliczane metryki:
    - accuracy, czyli dokładność,
    - sensitivity / recall, czyli czułość,
    - precision, czyli precyzja,
    - F1 score,
    - confusion matrix, czyli macierz pomyłek.
    """

    print("\n" + "=" * 80)
    print(f"ALGORYTM: {model_name}")
    print("=" * 80)

    print("Trenowanie modelu...")
    model.fit(X_train, y_train)
    print("Model został wytrenowany.")

    print("Wykonywanie predykcji na zbiorze testowym...")
    y_pred = model.predict(X_test)
    print("Predykcja została wykonana.")

    print("Obliczanie metryk jakości...")

    accuracy = accuracy_score(y_test, y_pred)

    # Dla klasyfikacji binarnej czułość liczona jest dla klasy pozytywnej.
    # W tym przykładzie klasą pozytywną jest 1.
    sensitivity = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print("\nWyniki:")
    print(f"Dokładność accuracy:     {accuracy:.4f}")
    print(f"Czułość recall:          {sensitivity:.4f}")
    print(f"Precyzja precision:      {precision:.4f}")
    print(f"F1 score:                {f1:.4f}")

    print("\nMacierz pomyłek:")
    print("Wiersze: klasy rzeczywiste")
    print("Kolumny: klasy przewidziane")
    print(pd.DataFrame(
        cm,
        index=["Rzeczywista 0", "Rzeczywista 1"],
        columns=["Przewidziana 0", "Przewidziana 1"]
    ))

    # Dla problemu binarnego można dodatkowo opisać elementy macierzy pomyłek.
    tn, fp, fn, tp = cm.ravel()

    print("\nInterpretacja macierzy pomyłek:")
    print(f"TN - poprawnie rozpoznane zera:        {tn}")
    print(f"FP - zera błędnie uznane za jedynki:   {fp}")
    print(f"FN - jedynki błędnie uznane za zera:   {fn}")
    print(f"TP - poprawnie rozpoznane jedynki:     {tp}")

    return {
        "Algorytm": model_name,
        "Accuracy": accuracy,
        "Czułość / Recall": sensitivity,
        "Precyzja": precision,
        "F1 score": f1
    }


# ============================================================
# 6. TRENOWANIE I OCENA WSZYSTKICH MODELI
# ============================================================

print("\nKROK 5: Trenowanie i ocena wszystkich algorytmów")

results = []

for model_name, model in models.items():
    result = evaluate_model(
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )
    results.append(result)


# ============================================================
# 7. PORÓWNANIE ALGORYTMÓW
# ============================================================

print("\n" + "=" * 80)
print("PODSUMOWANIE WYNIKÓW")
print("=" * 80)

results_df = pd.DataFrame(results)

# Sortujemy algorytmy od najlepszego do najgorszego według F1 score.
results_df = results_df.sort_values(by="F1 score", ascending=False)

print("\nTabela porównawcza algorytmów:")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]

print("\nNajlepszy algorytm według F1 score:")
print(f"{best_model['Algorytm']}")
print(f"F1 score: {best_model['F1 score']:.4f}")

print("\nKONIEC PROGRAMU")
print("=" * 80)