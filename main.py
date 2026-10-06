from pathlib import Path
import pandas as pd
from src.evaluation import (
    save_classification_reports,
    save_confusion_matrices,
    save_decision_tree_validation_curve,
    save_learning_curve_charts,
    save_model_comparison_charts,
    save_roc_curves,
)
from src.model import get_model_estimators, make_model_pipeline, train_and_evaluate_models
from src.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_data,
    load_data,
    prepare_features
)

def main():
    data_path = Path("data/heart_disease_dataset.csv")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result model").mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    df_clean = clean_data(df)
    X, y, numeric_features, categorical_features, binary_features = prepare_features(df_clean)
    trained_models, results, X_train, y_train, X_test, y_test = train_and_evaluate_models(
        X,
        y,
        numeric_features,
        categorical_features,
        binary_features,
    )
    results_df = pd.DataFrame(results).sort_values(by="Test F1", ascending=False)
    results_df.to_csv(output_dir / "result model"/"model_results.csv", index=False)
    save_model_comparison_charts(results_df, output_dir)
    save_confusion_matrices(trained_models, X_test, y_test, output_dir)
    save_classification_reports(trained_models, X_test, y_test, output_dir)
    save_roc_curves(trained_models, X_test, y_test, output_dir)
    save_learning_curve_charts(
        get_model_estimators(),
        make_model_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
        binary_features,
        output_dir,
    )
    save_decision_tree_validation_curve(
        make_model_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
        binary_features,
        output_dir,
    )
    print(f"\nĐã lưu kết quả vào thư mục {output_dir}/")
if __name__ == "__main__":
    main()
