from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import evaluate_model
from src.preprocess import build_preprocessor

#Khai báo các mô hình dùng bài 
def get_model_estimators():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="liblinear",
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_split=8 ),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        ),
    }

#Ghép tiền xử lý và mô hình thành Pipeline 
def make_model_pipeline(estimator, numeric_features, categorical_features, binary_features):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features, binary_features),
            ),
            ("model", clone(estimator)),
        ]
    )

#Chia dữ liệu thành train/test 
def split_dataset(X, y):
 
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

#Đánh giá mô hình bằng StratifiedKFold trên tập train
def cross_validate_model(
    model_name,
    estimator,
    X_train,
    y_train,
    numeric_features,
    categorical_features,
    binary_features,
):
    print(f"Đang đánh giá StratifiedKFold: {model_name}")
    model = make_model_pipeline(
        estimator,
        numeric_features,
        categorical_features,
        binary_features,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    return {
        "CV Accuracy Mean": scores["test_accuracy"].mean(),
        "CV Accuracy Std": scores["test_accuracy"].std(),
        "CV Precision Mean": scores["test_precision"].mean(),
        "CV Recall Mean": scores["test_recall"].mean(),
        "CV F1 Mean": scores["test_f1"].mean(),
        "CV F1 Std": scores["test_f1"].std(),
        "CV ROC AUC Mean": scores["test_roc_auc"].mean(),
        "CV ROC AUC Std": scores["test_roc_auc"].std(),
    }

#Huấn luyện từng mô hình một 
def train_single_model(
    model_name,
    estimator,
    X_train,
    y_train,
    numeric_features,
    categorical_features,
    binary_features,
):
    print(f"Đang huấn luyện: {model_name}")
    model = make_model_pipeline(
        estimator,
        numeric_features,
        categorical_features,
        binary_features,
    )
    model.fit(X_train, y_train)
    return model

#Huấn luyện tất cả mô hình, tạo bảng kết quả model
def train_and_evaluate_models(X, y, numeric_features, categorical_features, binary_features):
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    trained_models = {}
    results = []
    for model_name, estimator in get_model_estimators().items():
        cv_result = cross_validate_model(
            model_name,
            estimator,
            X_train,
            y_train,
            numeric_features,
            categorical_features,
            binary_features,
        )
        model = train_single_model(
            model_name,
            estimator,
            X_train,
            y_train,
            numeric_features,
            categorical_features,
            binary_features,
        )
        trained_models[model_name] = model
        result = evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
        result.update(cv_result)
        results.append(result)

    return trained_models, results, X_train, y_train, X_test, y_test
