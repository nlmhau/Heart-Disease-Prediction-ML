# Dự đoán Bệnh Tim (Heart Disease Prediction)

Dự án xây dựng mô hình dự đoán nguy cơ mắc bệnh tim từ các chỉ số lâm sàng (UCI/Kaggle Heart Disease). Mục tiêu: tạo pipeline hoàn chỉnh từ tiền xử lý đến huấn luyện, đánh giá và lưu mô hình để suy luận (inference).

## 1) Tổng quan nhanh
- Dữ liệu: 918 bệnh nhân, 11 biến đầu vào + 1 biến mục tiêu `HeartDisease` (0/1)
- Trọng tâm y tế: Ưu tiên giảm bỏ sót bệnh nhân (giảm False Negative) → nhấn mạnh Recall/ROC-AUC
- Kết quả chính (test 30%):
  - Logistic Regression: Accuracy ≈ 88.4%, Recall (class 1) ≈ 0.92, AUC > 0.90
  - Random Forest (tuned): Accuracy ≈ 87.7%, Recall ≈ 0.90, AUC > 0.90
  - SVM (RBF): Accuracy ≈ 87.3%, Recall ≈ 0.90, AUC > 0.90

## 2) Cấu trúc dự án
- `Heart_Disease_Prediction_ML.ipynb`: Notebook chính (EDA → Tiền xử lý → Huấn luyện → Đánh giá → Lưu model)
- `heart.csv`: Bộ dữ liệu gốc
- `saved_models/`: Thư mục chứa mô hình và tiện ích đã lưu bằng `joblib`
  - `heart_disease_rf_model.pkl`: Mô hình Random Forest sau tuning
  - `scaler.pkl`: StandardScaler đã fit trên tập train
  - `feature_columns.pkl`: Danh sách cột (thứ tự features khi inference)
- `README.md`: Tài liệu hướng dẫn (file này)
- `requirements.txt`: Danh sách thư viện Python tối thiểu

## 3) Quy trình xử lý (Data Pipeline)
1. Nạp dữ liệu và kiểm tra tổng quan (dtype, thiếu dữ liệu, phân phối lớp)
2. Làm sạch dữ liệu:
   - `RestingBP=0`, `Cholesterol=0` → thay thế bằng median (vì 0 là bất khả thi về sinh lý và dữ liệu có outliers)
   - Giữ nguyên `Oldpeak` âm (có ý nghĩa lâm sàng)
3. EDA: mô tả, boxplot, heatmap tương quan, phân tích tỷ lệ theo biến phân loại
4. Mã hóa biến phân loại:
   - Label Encoding: `Sex`, `ExerciseAngina`
   - One-Hot Encoding: `ChestPainType`, `RestingECG`, `ST_Slope`
5. Chia dữ liệu và chuẩn hóa:
   - `train_test_split` (70/30, `stratify=y`)
   - `StandardScaler` fit trên TRAIN, transform TEST (tránh data leakage)
6. Huấn luyện mô hình: Logistic Regression, Random Forest, SVM (RBF)
7. Đánh giá: Accuracy, Precision/Recall/F1, Confusion Matrix, ROC-AUC, 5-fold Cross-Validation
8. Tuning: `GridSearchCV` cho Random Forest (tối ưu theo Recall)
9. Lưu mô hình + scaler + danh sách cột bằng `joblib`

## 4) Cài đặt và chạy (Windows)
### a) Tạo môi trường và cài thư viện
```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### b) Mở và chạy notebook
- Mở VS Code/Notebook và chạy tuần tự các cell trong `Heart_Disease_Prediction_ML.ipynb`
- Hoặc khởi chạy Jupyter:
```bash
jupyter lab
```

## 5) Suy luận (Inference) với mô hình đã lưu
Ví dụ Python tối giản để dự đoán 1 bệnh nhân mới:
```python
import joblib
import pandas as pd

# Load
model = joblib.load('saved_models/heart_disease_rf_model.pkl')
scaler = joblib.load('saved_models/scaler.pkl')
cols = joblib.load('saved_models/feature_columns.pkl')

# Ví dụ dữ liệu mới (điền các giá trị phù hợp thực tế)
sample = pd.DataFrame([{
    'Age': 55,
    'Sex': 'M',
    'ChestPainType': 'ASY',
    'RestingBP': 140,
    'Cholesterol': 250,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 130,
    'ExerciseAngina': 'N',
    'Oldpeak': 1.0,
    'ST_Slope': 'Flat'
}])

# Encode giống pipeline trong notebook
sample['Sex'] = sample['Sex'].map({'F':0,'M':1})
sample['ExerciseAngina'] = sample['ExerciseAngina'].map({'N':0,'Y':1})
sample = pd.get_dummies(sample, columns=['ChestPainType','RestingECG','ST_Slope'], drop_first=True)

# Bổ sung các cột còn thiếu và sắp xếp đúng thứ tự
for c in cols:
    if c not in sample.columns:
        sample[c] = 0
sample = sample[cols]

# Scale các cột số (scaler đã học từ train)
# Lưu ý: trong notebook, scaler được áp dụng cho toàn bộ cột số trong X_train
numeric_cols = sample.select_dtypes(include=['int64','float64']).columns
sample[numeric_cols] = scaler.transform(sample[numeric_cols])

# Dự đoán
proba = model.predict_proba(sample)[0][1]
pred  = int(proba >= 0.5)
print('HeartDisease =', pred, ' | Probability =', round(proba*100, 2), '%')
```

## 6) Kết quả tóm tắt
- Tất cả mô hình đạt ROC-AUC > 0.90 (khả năng phân biệt rất tốt)
- Logistic Regression có Recall lớp mắc bệnh ≈ 0.92 → phù hợp mục tiêu y tế (giảm bỏ sót)
- Random Forest sau tuning cải thiện nhẹ Recall và cung cấp Feature Importance để giải thích (ST_Slope, MaxHR, Oldpeak nổi bật)

## 7) Phân công nhiệm vụ
- Nguyễn Lê Minh Hậu: Xử lý dữ liệu & tiền xử lý, xây dựng pipeline, áp dụng Logistic Regression, Random Forest và hỗ trợ viết báo cáo.
- Nguyễn Đức Huy: Hỗ trợ phân tích dữ liệu & xử lý dữ liệu, áp dụng SVM, tổng hợp kết quả và viết báo cáo.
- Bình Minh: không tham gia.

Mức độ tham gia: Hai thành viên cùng thảo luận và hoàn thiện bài theo tiến độ môn học.

## 8) Hạn chế & hướng phát triển
- Hạn chế: dữ liệu 918 mẫu còn nhỏ; chưa có external validation; chưa tuning threshold cho tối ưu Recall.
- Hướng phát triển: thử thêm XGBoost/LightGBM; threshold tuning theo Precision-Recall; calibration xác suất; feature selection (RFE/SelectKBest); bổ sung biến lâm sàng (BMI, hút thuốc, thuốc điều trị) và dữ liệu đa trung tâm.

## 9) Ghi chú sử dụng
- Mô hình chỉ hỗ trợ sàng lọc ban đầu, không thay thế chẩn đoán của bác sĩ chuyên khoa.

> Mặc dù pipeline và mô hình đã được xây dựng đầy đủ, trong báo cáo lần 1, nhóm tập trung trình bày các bước khám phá dữ liệu, tiền xử lý và kết quả ban đầu. Các nội dung nâng cao sẽ được phân tích chi tiết hơn trong báo cáo tiếp theo.
