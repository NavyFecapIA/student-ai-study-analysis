import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from dataset_generator import generate_synthetic_student_data


def load_or_create_dataset(csv_path: str = "student_study_performance.csv") -> pd.DataFrame:
    """Carrega o dataset se existir, caso contrário gera um novo."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Dataset carregado de '{csv_path}' com {len(df)} linhas.")
        return df

    print("Arquivo CSV não encontrado. Gerando novo dataset sintético...")
    df = generate_synthetic_student_data()
    df.to_csv(csv_path, index=False)
    print(f"Novo dataset salvo em '{csv_path}'.")
    return df


def perform_basic_eda(df: pd.DataFrame) -> None:
    """Realiza EDA básica: info, estatísticas, correlação e gráficos."""
    print("\n=== Informações do Dataset ===")
    print(df.info())

    print("\n=== Estatísticas Descritivas ===")
    print(df.describe())

    print("\n=== Matriz de Correlação ===")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix)

    # Visualização: mapa de calor da correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação - Student Study Performance")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=120)
    plt.close()

    # Dispersão: horas de estudo vs nota final
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="study_hours_per_day",
        y="final_exam_score",
        alpha=0.7,
    )
    plt.title("Horas de Estudo por Dia vs Nota Final")
    plt.xlabel("Horas de estudo por dia")
    plt.ylabel("Nota final no exame")
    plt.tight_layout()
    plt.savefig("study_hours_vs_score.png", dpi=120)
    plt.close()


def split_features_target(
    df: pd.DataFrame, target_col: str = "final_exam_score"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo."""
    feature_cols = [
        "study_hours_per_day",
        "sleep_hours",
        "social_media_hours",
        "practice_exams_completed",
        "class_attendance_percentage",
    ]
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[LinearRegression, RandomForestRegressor]:
    """Treina modelos de Regressão Linear e Random Forest."""
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_train)

    return lin_reg, rf_reg


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Avalia um modelo com MAE e R² e imprime resultados."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n=== Resultados do modelo: {name} ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² score: {r2:.3f}")


def plot_feature_importance(
    model: RandomForestRegressor, feature_names: list[str]
) -> None:
    """Plota a importância das features para o Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title("Importância das Features - Random Forest")
    plt.xlabel("Importância relativa")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("feature_importance_random_forest.png", dpi=120)
    plt.close()


def show_example_predictions(
    models: dict, feature_names: list[str]
) -> None:
    """Mostra previsões para estudantes de exemplo usando ambos modelos."""
    example_students = pd.DataFrame(
        [
            {
                "study_hours_per_day": 2.0,
                "sleep_hours": 6.5,
                "social_media_hours": 3.0,
                "practice_exams_completed": 3,
                "class_attendance_percentage": 75.0,
            },
            {
                "study_hours_per_day": 4.5,
                "sleep_hours": 7.5,
                "social_media_hours": 1.0,
                "practice_exams_completed": 8,
                "class_attendance_percentage": 95.0,
            },
            {
                "study_hours_per_day": 1.0,
                "sleep_hours": 5.5,
                "social_media_hours": 5.0,
                "practice_exams_completed": 0,
                "class_attendance_percentage": 60.0,
            },
        ],
        columns=feature_names,
    )

    print("\n=== Estudantes de Exemplo ===")
    print(example_students)

    for name, model in models.items():
        preds = model.predict(example_students)
        print(f"\nPrevisões de nota final usando {name}:")
        for i, pred in enumerate(preds, start=1):
            print(f"  Estudante {i}: nota prevista = {pred:.1f}")


def main() -> None:
    """Fluxo principal: EDA, treino, avaliação, visualizações e previsões."""
    df = load_or_create_dataset()

    perform_basic_eda(df)

    X, y = split_features_target(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    lin_reg, rf_reg = train_models(X_train, y_train)

    evaluate_model("Linear Regression", lin_reg, X_test, y_test)
    evaluate_model("Random Forest Regressor", rf_reg, X_test, y_test)

    plot_feature_importance(rf_reg, feature_names)

    models = {
        "Linear Regression": lin_reg,
        "Random Forest Regressor": rf_reg,
    }
    show_example_predictions(models, feature_names)


if __name__ == "__main__":
    main()

