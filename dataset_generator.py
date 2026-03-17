import numpy as np
import pandas as pd


def generate_synthetic_student_data(
    n_samples: int = 300, random_state: int = 42
) -> pd.DataFrame:
    """Gera um conjunto de dados sintético de estudantes.

    As relações abaixo são apenas aproximações realistas, criadas para fins educacionais.
    """
    rng = np.random.default_rng(seed=random_state)

    # Características principais (features)
    study_hours_per_day = rng.normal(loc=3.5, scale=1.2, size=n_samples)
    study_hours_per_day = np.clip(study_hours_per_day, 0.5, 8.0)

    sleep_hours = rng.normal(loc=7.0, scale=1.0, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)

    social_media_hours = rng.normal(loc=2.5, scale=1.5, size=n_samples)
    social_media_hours = np.clip(social_media_hours, 0.0, 8.0)

    practice_exams_completed = rng.integers(low=0, high=16, size=n_samples)

    class_attendance_percentage = rng.normal(loc=85.0, scale=10.0, size=n_samples)
    class_attendance_percentage = np.clip(class_attendance_percentage, 40.0, 100.0)

    # Construção da nota final com uma combinação linear + ruído
    base_score = 20
    score_from_study = study_hours_per_day * 7.0
    score_from_sleep = (sleep_hours - 6.0) * 2.0
    score_from_social = -social_media_hours * 1.5
    score_from_practice = practice_exams_completed * 1.8
    score_from_attendance = (class_attendance_percentage / 100.0) * 15.0

    noise = rng.normal(loc=0.0, scale=8.0, size=n_samples)

    final_exam_score = (
        base_score
        + score_from_study
        + score_from_sleep
        + score_from_social
        + score_from_practice
        + score_from_attendance
        + noise
    )
    final_exam_score = np.clip(final_exam_score, 0.0, 100.0)

    data = pd.DataFrame(
        {
            "study_hours_per_day": study_hours_per_day,
            "sleep_hours": sleep_hours,
            "social_media_hours": social_media_hours,
            "practice_exams_completed": practice_exams_completed,
            "class_attendance_percentage": class_attendance_percentage,
            "final_exam_score": final_exam_score,
        }
    )

    return data


def main() -> None:
    """Gera e salva o dataset sintético em CSV."""
    df = generate_synthetic_student_data(n_samples=300, random_state=42)
    output_path = "student_study_performance.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset sintético salvo em '{output_path}' com {len(df)} linhas.")


if __name__ == "__main__":
    main()

