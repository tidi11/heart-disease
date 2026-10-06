from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
SELECTED_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

#Tạo OneHotEncoder tương thích với nhiều phiên bản scikit-learn.
def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

#Đọc dữ liệu từ file CSV 
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path}")
    df = pd.read_csv(path)
    print(f"Kích thước dữ liệu ban đầu: {df.shape}")
    return df

#Xóa dòng trùng và đưa giá trị bất thường ca=4 về ca=3.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna()
    df_clean.loc[df_clean["ca"] == 4, "ca"] = 3
    print(f"Kích thước dữ liệu sau làm sạch: {df_clean.shape}")
    return df_clean

 #Tách dữ liệu thành X là đặc trưng đầu vào và y là nhãn cần dự đoán
def prepare_features(df: pd.DataFrame):
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y, NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES

#Chuẩn hóa biến số, one-hot biến phân loại và giữ nguyên biến nhị phân
def build_preprocessor(numeric_features, categorical_features, binary_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
            ("bin", "passthrough", binary_features),
        ]
    )
#Fit preprocessor
def preprocess_data(df: pd.DataFrame):
    X, y, numeric_features, categorical_features, binary_features = prepare_features(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features, binary_features)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    return X_processed, y, preprocessor
