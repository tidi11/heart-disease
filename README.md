# MÔ HÌNH CLASSIFICITION
## Chủ đề: Heart dsiease

### 1. Mục tiêu 
Dự đoán một người có bị bệnh tim không, so sánh và phân tính đánh giá các mô hình, đề xuất hướng cải thiện
---

## 2. Cấu trúc thư mục
```text
.
├── README.md
├── data/
│   └── heart_disease_dataset.csv
├── eda_exploration.ipynb
├── main.py
├── outputs/
│   ├── eda/
│   └── result model/
├── report.md
├── requirements.txt
└── src/
    ├── evaluation.py
    ├── model.py
    └── preprocess.py
```
Ý nghĩa các file:
- `README.md`: tài liệu hướng dẫn 
- `report.md`: nhận xét kết quả cuối bài.
- `requirements.txt`: danh sách thư viện cần cài đặt.
- `main.py`: file chạy chính.
- `data/heart_disease_dataset.csv`: dữ liệu Loan Approval Prediction.
- `eda_exploration.ipynb`: trình bày bước khám phá dữ liệu
- `src`: mã nguồn xử lý dữ liệu, huấn luyện, đánh giá và vẽ biểu đồ.
- `outputs/`: thư mục chứa kết quả sau khi chạy.


---
## 3. Chuẩn bị môi trường

Tạo môi trường ảo nếu cần
```bash
python -m venv venv
```

Kích hoạt môi trường ảo:
- Trên Windows:
  ```bash
  venv\Scripts\activate
  ```
- Trên macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

Cài đặt các thư viện cần thiết
Chạy lệnh sau để cài đặt các thư viện được liệt kê trong file `requirements.txt`:
```bash
pip install -r requirements.txt
```

---
## 4. Giới thiệu bài toán
Dự đoán bệnh nhân có mắc bệnh tim hay không
Biến mục tiêu:
``` text
target
```
Các lớp:
0: không mắc bệnh tim
1: có mắc bệnh tim 

Các đặc trưng trong dataset:
- age   : Tuổi của bệnh nhân tính theo năm.
- sex   : Giới tính
    + 1 : Nam
    + 0 : Nữ 
- cp    : Loại đau ngực
    + 0 : Đau thắt ngực điển hình.
    + 1 : Đau thắt ngực không điển hình.
    + 2 : Đau ngực không phải do thắt ngực.
    + 3 : Không có triệu chứng đau ngực.
- trestbps : Huyết áp lúc nghỉ
- chol  : Lượng Cholesterol trong huyết thanh 
- fbs	: Lượng đường trong máu lúc đói
    + 1: Lớn hơn 120 mg/dl (Đường huyết cao)
    + 0: Nhỏ hơn hoặc bằng 120 mg/dl (Bình thường).
- restecg	: Kết quả điện tâm đồ lúc nghỉ
    + 0 = Bình thường.
    + 1 = Có sóng ST-T bất thường 
    + 2 = Phì đại thất trái có khả năng hoặc chắc chắn.
- thalach	: Nhịp tim tối đa
- exang	: Đau thắt ngực do gắng sức/tập thể dục
    + 1: Có đau
    + 0: Không đau 
-  oldpeak	: Độ chênh xuống của đoạn ST (Đoạn ST chênh xuống càng sâu (chỉ số oldpeak càng cao) cho thấy tình trạng thiếu máu cơ tim càng nghiêm trọng)
- slope	: Độ dốc của đoạn ST khi gắng sức đỉnh điểm
    + 0 = Dốc xuống Nhịp tim khỏe mạnh.
    + 1 = Đi ngang Có vấn đề nhẹ.
    + 2 = Dốc lên Dấu hiệu bệnh lý về tim
- ca: số lượng các mạch máu chính bị tắc nghẽn hoặc tổn thương
- thal : Rối loạn máu Thalassemia
    + 1 :  Bình thường
    + 2 : Khiếm khuyết cố định
    + 3 : Khiếm khuyết có thể phục hồi 
- target: 
    + 1 : Có bệnh tim 
    + 0 : Không có bệnh tim

## 5. Các bước thực hiện

### Bước 1. Đọc dữ liệu
Đọc dữ liệu từ `data/heart_disease_datasetcsv`
### Bước 2. Khám phá dữ liệu EDA
Kiểm tra tổng quan và hiểu cấu trúc của dataset
Kiểm tra dữ liệu bị thiếu và bất thường
Phân tích từng nhóm categorical features và numerical
Phân tích mối quan hệ và tương quan của các biến với nhau, và với biến mục tiêu
### Bước 3: Làm sạch dữ liệu
Xử lý các dòng trùng lặp
Xử lý outlier: thay ca=4 thành ca=3
### Bước 4: Chuẩn bị các đặc trưng và chạy mô hình
Ta tách:

- `X`: các đặc trưng đầu vào. (thalach, fbs,  chol, trestbps, age, thal,cp,slope)
- `y`: nhãn cần dự đoán. (target)

Các cột số được chuẩn hóa bằng `StandardScaler`. (chol, age,trestbps, thalach )

Các cột phân loại được mã hóa bằng `OneHotEncoder` (cp, slope, thal)
###### Bước 5. Đánh giá mô hình
Báo cáo các chỉ số:

- Accuracy.
- Precision.
- Recall.
- F1-score.

Vẽ:

- Confusion Matrix.
- Classification report.
- ROC Curve và ROC-AUC 
- Learning curve hoặc validation curve 
### Bước 5. So sánh và phân tích kết quả

Bảng so sánh kết quả giữa các mô hình.

Phần phân tích cần trả lời:

- Mô hình nào cho kết quả tốt nhất?
- Dựa trên chỉ số nào để chọn mô hình tốt nhất?
- Vì sao mô hình đó có thể hoạt động tốt hơn các mô hình còn lại?
- Có dấu hiệu overfitting hoặc underfitting không?
- Những khó khăn gặp phải trong quá trình thực hiện.
- Hướng cải thiện mô hình trong tương lai.

---


