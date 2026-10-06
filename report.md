# Nhận xét kết quả cuối bài

## 1. Tổng quan kết quả

Sau khi chạy thực nghiệm trên tập heart_disease_dataset , các mô hình được đánh giá trên tập test bằng Accuracy, Precision, Recall, F1-score và ROC-AUC.

**Bảng kết quả khi tập test giữa nguyên :**
`outputs/result model/model_results.csv`


| Model | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train F1 | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.884 |	0.852 |	0.860 |	0.875 | 0.939 |	0.848 |	0.898 |	0.862 |	0.902|
| Random Forest |0.884 | 0.836 | 0.860 | 0.829 | 0.939 |0.879 |	0.898 |	0.853 |	0.903 |
| Decision Tree | 0.859 | 0.803 | 0.830 |0.784 | 0.931 | 0.879 | 0.878 | 0.829 | 0.813 |
| Naive Bayes | 0.859 |	0.803 |	0.854 |	0.818 |	0.893 |	0.818 |	0.873 | 0.818 |	0.885 |

Khi để nguyên tập test, nhìn chung, tất cả mô hình đều có kết quả khá tốt. Logistic Regression có hiệu quả hơn ở Accuracy, Precision,  F1. Tuy nhiên Random Forest, Decision Tree  lại có chỉ số Recall cao hơn. Và Random Forest có chỉ số ROC-AUC cao nhất.

![ROC Curves](outputs/result%20model/roc_curves.png)

**Bảng kết quả khi chia nhỏ tập test và tính trung bình:**
| Model | CV Accuracy Mean | CV Accuracy Std | CV Precision Mean | CV Recall Mean | CV F1 Mean | CV F1 Std | CV ROC AUC Mean | CV ROC AUC Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.838 |	0.084 |	0.842 |	0.885 |	0.856 |	0.071 |	0.913 | 0.045 |
| Random Forest |0.863 | 0.060 | 0.852 | 0.915 | 0.879 | 0.050 | 0.914 | 0.046 |
| Decision Tree | 0.780 | 0.038 | 0.795	| 0.809 | 0.800	| 0.034 | 0.813 | 0.044 |
| Naive Bayes | 0.767 |	0.161 |	0.816 |	0.816 |	0.787 |	0.145 |	0.852 |	0.079 |

Tuy nhiên khi chia nhỏ tập test ra và tính trung bình , Random Forest lại là mô hình có các chỉ số cao hơn rõ rệt. Logistic Regression còn chỉ đứng thứ nhì 

Nếu xét theo cross-validation thì Random Forest tổng quát tốt hơn. Chỉ số Recall( được ưu tiên trong bài toán này) của Random Forest cũng cao hơn rõ rệt.  Đồng thời độ lệch chuẩn (std) của các chỉ số của Random Forest cũng nhỏ hơn của Logistic Regression, nên Random Forest cực kỳ ổn định, ít bị dao động ngay cả khi dữ liệu đầu vào thay đổi. Vì vậy ta ưu tiên lựa chọn mô hình Random Forest.

![Model Comparison CV Accuracy mean](outputs/result%20model/model_comparison_cv_accuracy_mean.png)
![Model Comparison CV F1 mean](outputs/result%20model/model_comparison_cv_f1_mean.png)


## 2. So sánh từng mô hình

### Logistic Regression
 
Kết quả:
- `CV Accuracy Mean = 0.838`
- `CV Accuracy Std = 0.084`
- `CV Precision Mean = 0.842`
- `CV Recall Mean = 0.885`
- `CV F1 Mean = 0.856`
- `CV F1 Std = 0.071`
- `CV ROC AUC Mean = 0.913`
- `CV ROC AUC Std = 0.045`

Diễn giải: 
- Accuracy Mean = 0.838 cho thấy Logistic Regression đoán được hầu hết trên tập dữ liệu. 
- Accuracy Std = 0.084 là biên độ khá rộng. Điều này có nghĩa là ở một số tập dữ liệu con, mô hình có thể dự đoán chính xác trên 92%, nhưng ở một tập khác lại rớt xuống chỉ còn khoảng 75%. Mô hình không đủ sự ổn định
- Nguyên nhân do Logistic Regression nhạy cảm với outlier. Ở bước EDA, ta nhận thấy bộ dữ liệu này có nhiều outlier, nhưng vì đây là dữ liệu y tế, những oulier có thể là cảnh báo của triệu chứng nặng nên t đã k chọn loại bỏ. Vì vậy  các outliers gây xô lệch đường ranh giới quyết định
- ROC AUC Mean = 0.913: là một số điểm rất tốt. Chứng tỏ mô hình học được bản chất của dữ liệu để phân biệt hai lớp ( bị bệnh- không bị bệnh) rất rõ ràng.
- Recall Mean (0.885) cao hơn CV Precision Mean (0.842): do bài toán đang ưu tiên phát hiện các bệnh nhân bị bệnh tim, nên mô hình này đang làm tốt mục đích 

Kết luận: mô hình dự đoán tốt nhưng lại thiếu sự ổn định, phụ thuộc vào cách chia tập dữ liệu 
### Decision Tree với `max_depth=3`
Kết quả:
- `CV Accuracy Mean = 0.780`
- `CV Accuracy Std = 0.038`
- `CV Precision Mean = 0.795`
- `CV Recall Mean = 0.809`
- `CV F1 Mean = 0.800`
- `CV F1 Std = 0.034`
- `CV ROC AUC Mean = 0.813`
- `CV ROC AUC Std = 0.044`

Diễn giải: 
- Accuracy Mean = 0.780 khá thấp nhưng chưa phải thấp nhất. Cho thấy mô hình chỉ dự đoán được phần nào tập dữ liệu, 
- Recall Mean = 0.809 thấp nhất trong các mô hình. Decision Tree bỏ sót quá nhiều bệnh nhân bị bệnh tim. Không nên sử dụng mô hình này
- CV Accuracy Std = 0.038 và CV F1 Std = 0.034: tuy nhiên mô hình lại có sự ổn định, ít bị dao động bởi các lần chia tập dữ liệu. 

Kết luận: Mô hình ổn định nhưng dự đoán tổng thể lại không tốt. 

### Naive Bayes 
Kết quả:
- `CV Accuracy Mean = 0.767`
- `CV Accuracy Std = 0.161`
- `CV Precision Mean = 0.816`
- `CV Recall Mean = 0.816`
- `CV F1 Mean = 0.879`
- `CV F1 Std = 0.050`
- `CV ROC AUC Mean = 0.914`
- `CV ROC AUC Std = 0.046`

Diễn giải :
- Accuracy Std = 0.161 và CV F1 Std = 0.145: đây là độ lệch chuẩn rất lớn. Hiệu suất của mô hình rất thất thường 
- Recall Mean = 0.816: cũng thấp như Decision Tree. Không thể sử dụng mô hình này vào bộ dữ liệu này 

Kết luận: đây là mô hình cho kết quả dự đoán yếu nhất. 

### Random Forest
Kết quả:
- `CV Accuracy Mean = 0.863`
- `CV Accuracy Std = 0.060`
- `CV Precision Mean = 0.852`
- `CV Recall Mean = 0.915`
- `CV F1 Mean = 0.878`
- `CV F1 Std = 0.053`
- `CV ROC AUC Mean = 0.914`
- `CV ROC AUC Std = 0.046`

Diễn giải:
- Accuracy Mean = 0.863: trong số các mô hình, đây là mô hình dự đoán đúng được nhiều nhất trên tập dữ liệu 
- Recall Mean = 0.915: không bỏ sót qua nhiều bệnh nhân, tốt nhất trong tất cả các mô hình 
- Accuracy Std = 0.060 và ROC AUC Std = 0.046 : mô hình cho ra kết quả dự báo ổn định 
- CV ROC AUC Mean = 0.914: mô hình có khả năng phân tách rạch ròi giữa nhóm người bệnh và người khỏe mạnh

Kết luận: mô hình vừa đảm bảo được khả năng dự đoán đúng, vừa đủ sự ổn định, không bỏ sót quá nhiều bệnh nhân. Nên đây là mô hình tốt nhất



## 3. Nhận xét confusion matrix
Dựa trên classification report, tập test có:
- 33 Bệnh tim
- 28 Không bị bệnh tim

Với mô hình  Random Forest
| Thực tế | Dự đoán bị bệnh | Dự đoán không bị bệnh |
|---|---:|---:|
| Bị bệnh | 29 | 4 |
| Không bị bệnh | 6 | 22 |

![Confusion Matrix Random Forest](outputs/result%20model/confusion_matrix_random_forest.png)

Diễn giải:
- True Positive (TP) = 29: mô hình dự đoán đúng 29 ngừoi bị `bệnh tim`
- True Negative (TN) = 22: mô hình dự đoán đúng 22 ngừoi `khoẻ mạnh`
- False Positive (FP) = 6: Có 6 người `khỏe mạnh`, nhưng mô hình lại dự đoán nhầm là họ có `bệnh tim`.
- False Negative (FN) = 4: Có 4 bệnh nhân thực sự có `bệnh tim` , nhưng mô hình lại dự đoán sai là họ `khỏe mạnh`.

Trong bối cảnh y tế(phát hiện bệnh tim):
- Mối nguy hiểm của sai lầm False Negative (FN) (bỏ lọt ca bệnh) là nguy hiểm nhất. 4 bệnh nhân này có vấn đề về tim mạch, nhưng mô hình lại báo họ khỏe mạnh. Sai lầm này có thể dẫn tới các biến chứng nguy hiểm vì không được phát hiện
- False Positive(FP) là sai lầm có thể chấp nhận được, vì chỉ gây các thiệt hại về thời gian và tiền bạc để kiểm tra sâu hơn. Chứ không tạo ra nguy cơ nguy hiểm đến tính mạng

Mục tiêu: 
- Giảm sai lầm FP xuống thấp nhất có thể bằng cách tăng chỉ số Recall lên


Với mô hình Logistic Regression
| Thực tế | Dự đoán bị bệnh | Dự đoán không bị bệnh |
|---|---:|---:|
| Bị bệnh | 28 | 5 |
| Không bị bệnh | 4 | 24 |

![Confusion Matrix Logistic Regression](outputs/result%20model/confusion_matrix_logistic_regression.png)

Với mô hình Decision Tree
| Thực tế | Dự đoán bị bệnh | Dự đoán không bị bệnh |
|---|---:|---:|
| Bị bệnh | 29 | 4 |
| Không bị bệnh | 8 | 20 |

![Confusion Matrix Decision Tree](outputs/result%20model/confusion_matrix_decision_tree.png)

Với mô hình Naive Bayes
| Thực tế | Dự đoán bị bệnh | Dự đoán không bị bệnh |
|---|---:|---:|
| Bị bệnh | 27 | 6 |
| Không bị bệnh | 6 | 22 |

![Confusion Matrix Naive Bayes](outputs/result%20model/confusion_matrix_naive_bayes.png)

## 4. Nhận xét learning curve

Learning curve cho thấy sự thay đổi hiệu năng khi tăng dần số lượng mẫu train.

### Logistic Regression
| Train Size | Train F1 | CV Val F1 | Train Accuracy |CV Val Accuracy |
|---:|---:|---:|---:|---:|
| 28 |0.968 | 0.798 | 0.964 |0.772 |
| 52 |0.967 | 0.856 | 0.962 | 0.834 |
| 75 | 0.904 | 0.838 | 0.901 | 0.817 |
| 98 |0.870 | 0.8452 | 0.861 | 0.821 |
| 122 |0.873 | 0.842 | 0.866 | 0.821 |
| 145 |0.875 | 0.851 | 0.865 | 0.834 |
| 168 |0.884 | 0.852 | 0.876 | 0.834 |
| 192 |0.891 | 0.861 | 0.877 | 0.842 |

![Learning Curve Logistic Regression Accuracy](outputs/result%20model/learning_curve_logistic_regression_accuracy.png)
![Learning Curve Logistic Regression F1](outputs/result%20model/learning_curve_logistic_regression_f1.png)

Nhận xét:
- Khi tập train <50, mô hình có dấu hiệu của overfitting nhẹ 
- Kết quả tăng khi tăng dữ liệu. Hai đường bắt đầu thu hẹp khoảng cách. 
- Tuy nhiên từ mốc 100 trở đi, khoảng cách giữa hiệu suất trên tập train và tập test duy trì ở mức rất nhỏ và gần như không đổi. Mô hình Logistic Regression đã khai thác hết thông tin.
- Logistic Regression hoàn toàn không bị overfitting trên bộ dữ liệu này

### Decision Tree

| Train Size | Train F1 | CV Val F1 | Train Accuracy |CV Val Accuracy |
|---:|---:|---:|---:|---:|
| 28 | 0.987 | 0.793 | 0.986 | 0.764 |
| 52 |0.954 | 0.760 | 0.946 |0.731 |
| 75 |0.938 | 0.760 | 0.934 | 0.734 |
| 98 |0.898 | 0.784 | 0.888 | 0.764 |
| 122 |0.864 | 0.739 | 0.862 | 0.722 |
| 145 |0.886 | 0.770 | 0.877 | 0.747 |
| 168 |0.870 | 0.717 | 0.867 | 0.710 |
| 192 | 0.878 | 0.813 | 0.866 | 0.793 |

![Learning Curve Decision Tree Accuracy](outputs/result%20model/learning_curve_decision_tree_accuracy.png)
![Learning Curve Decision Tree F1](outputs/result%20model/learning_curve_decision_tree_f1.png)

Nhận xét: 
- Khi tăng train size hiệu suất trên tập test không tăng đáng kể mà còn trồi sụt liên tục. Mô hình thiếu tính ổn định.
- Khoảng cách giữa hai đường vẫn lớn, không có dấu hiện tiến sát gần nhau 
- Có dấu hiệu của overfitting

### Random Forest

| Train Size | Train F1 | CV Val F1 | Train Accuracy |CV Val Accuracy |
|---:|---:|---:|---:|---:|
| 28 | 0.877 | 0.804 | 0.857 | 0.751 |
| 52 | 0.940 | 0.847 | 0.927 | 0.813 |
| 75| 0.905 | 0.852 | 0.899 | 0.834 |
| 98 | 0.866 | 0.841 | 0.855 | 0.825 |
| 122 | 0.874 | |0.828 | 0.862 | 0.809 |
| 145 | 0.881 | 0.849 | 0.868 | 0.830 |
| 168 | 0.885 | 0.857 | 0.876 | 0.842 |
| 192 | 0.893 | 0.883 | 0.879 | 0.867 |

Nhận xét: 
- Khi tăng tập train tới gần 200, hai đường gần như chạm vào nhau. Khoảng cách được thu hẹp đến mức tối đa.
- Random Forest đã tổng quát hóa dữ liệu cực kỳ xuất sắc. Hiệu suất mô hình đạt được trên tập train (khoảng 0.89) gần như được giữ nguyên khi dự đoán trên dữ liệu tập test (khoảng 0.88).
- Hiệu suất tập test tăng ổn định. Mô hình đã đưa ra những dự đoán kiên định, đáng tin cậy
- Hai đường chưa có dấu hiệu đi ngang, ở đoạn kích thức tập train là 200, đường hiệu suất tập test vẫn đang có xu hướng hơi nhích lên nhẹ.
- Năng lực của mô hình Random forest lớn hơn Logistic Regression, vì nếu thu thập thêm dữ liệu Random forest vẫn còn có thể học thêm và cải thiện hiệu suất 
![Learning Curve Random Forest Accuracy](outputs/result%20model/learning_curve_random_forest_accuracy.png)
![Learning Curve Random Forest F1](outputs/result%20model/learning_curve_random_forest_f1.png)

### Naive Bayes   

| Train Size | Train F1 | CV Val F1 | Train Accuracy |CV Val Accuracy |
|---:|---:|---:|---:|---:|
| 28 | 0.897 | 0.815 | 0.886 | 0.805 |
| 52 | 0.847 | 0.757 | 0.85 | 0.759 | 
| 75 | 0.885 | 0.799 | 0.883 | 0.780 |
| 98 | 0.846 | 0.835 | 0.839 | 0.817 |
| 122 | 0.854 | 0.841 | 0.844 | 0.821 |
| 145 | 0.806 | 0.777 | 0.746 | 0.714 | 
| 168 | 0.791 | 0.785 | 0.770 | 0.763 |
| 192 | 0.807 | 0.787 | 0.783 | 0.767 |

- Khi tăng kích thước tập train, khoảng cách của hai đường dần thu hẹp. Hai đường chạm nhưng ở mức điểm thấp. Có dấu hiệu của underfitting
- Từ 125 đến 150, cả hai đường đều giảm mạnh. Mô hình cực kỳ nhạy cảm và hỗn loạn khi thêm dữ liệu mới, rất thiếu tính ổn định


![Learning Curve Naive Bayes Accuracy](outputs/result%20model/learning_curve_naive_bayes_accuracy.png)
![Learning Curve Naive Bayes F1](outputs/result%20model/learning_curve_naive_bayes_f1.png)

## 5. Nhận xét validation curve của Decision Tree

Validation curve được vẽ bằng cách thay đổi `max_depth`.

| max_depth | Train F1 | CV Val F1 | Train Accuracy | CV Val Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.801 | 0.778 | 0.785 | 0.759 |
| 2 | 0.817 | 0.809 | 0.791 | 0.788 |
| 3 | 0.883 | 0.800 | 0.868 | 0.780 |
| 4 | 0.913 | 0.794 | 0.902 | 0.763 |
| 5 | 0.960 | 0.755 | 0.957 | 0.730 |
| 7 | 0.998 | 0.769 | 0.998 | 0.747 |
| 10 | 1.0 | 0.773 | 1.0 | 0.755 |
| None | 1.0 | 0.773 | 1.0 | 0.755 | 

![Validation Curve Decision Tree Accuracy](outputs/result%20model/validation_curve_decision_tree_accuracy.png)
![Validation Curve Decision Tree F1](outputs/result%20model/validation_curve_decision_tree_f1.png)

Diễn giải:
- Khi cây quá nông, `max_depth`=1, cây bị underfitting
- Hai đường gần chạm nhau khi `max_depth`=2, cây vừa đủ phức tạp để tách biệt các nhóm bệnh nhân, nhưng cũng đủ đơn giản để không bị nhiễu. Đây là giá trị `max_depth` nên chọn
- Khi `max_depth` >=3, khoảng cách giữa hai đường tăng lên rất lớn, mô hình bị overfitting. 
- Với Decision Tree, `max_depth` nên được giới hạn thay vì để `None`.


## 6. Kết luận cuối bài

- Random forest là mô hình làm tốt nhất
    +  Độ nhạy Recall cao nhất trong các mô hình, tình trạng dự đoán bỏ sót bệnh nhân (FP) là thấp nhất theo confusion matrix là 4 người
    +  Mô hình đảm bảo tính ổn định vì độ lệch chuẩn std của các chỉ số đều thấp. Xem xét đường learing curve khi tăng kích thước tập dữ liệu, hiệu suất cũng tăng ổn định và nhất quán, không bị trồi sụt liên tục. Chúng giúp đảm bảo mô hình dự đoán nhất quán trên nhiều tập dữ liệu khác nhau.
    + ROC-AUC cao: mô hình có khả năng phân tách rạch ròi giữa nhóm người bệnh và người khỏe mạnh
    + F1-Score ở mức tốt, có sự cân bằng giữa chỉ số Precision và Recall


- Khó khăn:
    + Bộ dữ liệu có nhiều outlier, tuy nhiên để quyết định được đây là nhiễu hay chỉ số dự báo trình trạng bệnh nghiêm trọng còn khó. 
    + Việc  giữ lại các outlier này để bảo toàn thông tin y tế đã gây khó khăn cho các mô hình Logistic Regression.
    + Để phát hiện được nhiều bệnh nhân nhất (tăng Recall), thì sẽ phải đánh đổi điểm Precision giảm xuống (tỷ lệ chẩn đoán nhầm người khỏe thành có bệnh tăng lên)
    + Tập dữ liệu sau khi làm sạch còn rất nhỏ: khiến mô hình nhạy cảm như Naive Bayes có hiện tượng sụt giảm hiệu năng đột ngột

- Cải thiện: 
    +  Do ROC AUC Mean = 0.914 của Random Forset khá cao đồng thời ROC AUC Std cũng thấp. Nên ta có thể hạ ngưỡng quyết định (Threshold) để đẩy Recall lên cao hơn nữa  mà không làm chỉ số Precision giảm thấp mhiều. 
    + Tạo thêm nhiều feature dựa trên kiến thức y khoa để đưa vào mô hình thay vì chỉ dùng các chỉ số ban đầu, giúp mô hình bắt được các mẫu hình phức tạp hơn.
    + Với mô hình Logistic Regression nhạy cảm với các outlier, nhưng ta cũng không thể bỏ các dữ liệu này vì đây là dữ liệu y tế. Nên ta có thể  cho các outlier bằng 1 giá trị cao nhất, nhất định.
    + Thu thập thêm nhiều dữ liệu hơn sẽ giúp mô hình  tăng hiệu suất