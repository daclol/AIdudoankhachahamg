# Import thư viện cần thiết
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🔹 Bước 1: Đọc dữ liệu
file_path = r"c:/Users\Admin\Documents\Zalo Received Files/kiemtra\BTL Python\AI\Artificial_Neural_Network_Case_Study_data.csv"
  # Cập nhật đường dẫn của bạn
df = pd.read_csv(file_path)

# 🔹 Bước 2: Xử lý dữ liệu
# Loại bỏ cột không cần thiết
customer_info = df[['CustomerId', 'Surname']]  # Giữ lại thông tin khách hàng
X = df.drop(columns=['Exited', 'CustomerId', 'Surname', 'RowNumber'])  # Dữ liệu đầu vào
y = df['Exited']  # Nhãn cần dự đoán

# Mã hóa biến phân loại
X = pd.get_dummies(X, columns=['Gender', 'Geography'], drop_first=True)

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Bước 3: Chia dữ liệu thành tập huấn luyện & kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Bước 4: Xây dựng mô hình mạng neuron (ANN)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden Layer 1
    keras.layers.Dense(8, activation='relu'),  # Hidden Layer 2
    keras.layers.Dense(1, activation='sigmoid')  # Output Layer
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 Bước 5: Huấn luyện mô hình
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# 🔹 Bước 6: Dự đoán trên toàn bộ dữ liệu (10.000 khách hàng)
y_pred_prob_full = model.predict(X_scaled)  # Dự đoán xác suất
y_pred_binary_full = (y_pred_prob_full > 0.5).astype(int)  # Chuyển thành 0 hoặc 1

# 🔹 Bước 7: Xuất kết quả đầy đủ
results_full = customer_info.copy()  # Giữ lại thông tin khách hàng
results_full['Thực tế'] = y.values  # Giá trị thực tế
results_full['Dự đoán'] = y_pred_binary_full.flatten()  # Giá trị dự đoán

# Xuất toàn bộ dữ liệu ra file CSV
output_file_full = r"c:\Users\Admin\Documents\Zalo Received Files\kiemtra\BTL Python\AI\KetQuaDuDoan_Full.xlsx"
results_full.to_csv(output_file_full, index=False)
print(f" Kết quả đã được lưu vào file: {output_file_full}")

# 🔹 Bước 8: Xuất 10.000 khách hàng đầu tiên (nếu có đủ )
print(results_full.head(10000))

# 🔹 Bước 9: Lưu mô hình mạng neuro
model.save(r"c:\Users\Admin\Documents\Zalo Received Files\kiemtra\BTL Python\AI\model.h5")
print(" Mô hình mạng neuron đ đã được lưu vào file: Artificial_Neural_Network_Case_Study_model.h5")

# 🔹 Bước 10: Đọc mô hình mạng neuro
model = keras.models.load_model(r"c:\Users\Admin\Documents\Zalo Received Files\kiemtra\BTL Python\AI\model.h5")
print(" Mô hình mạng neuron đã đọc từ file: Artificial_Neural_Network_Case_Study_model.h5")

# 🔹 Bước 11: Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}")

# 🔹 Bước 12: Dự đoán trên toàn bộ dữ liệu (10.000 khách hàng)
y_pred_prob_full = model.predict(X_scaled)  # Dự đoán xác suất
y_pred_binary_full = (y_pred_prob_full > 0.5).astype(int)  # Chuyển thành 0 hoặc 1

# 🔹 Bước 13: Xuất kết quả đày đ
# Kết quả dự đoán theo đúng thứ tự ban đủ
results_full = customer_info.copy()
results_full['Thực tế'] = y.values
results_full['Dự đoán'] = y_pred_binary_full.flatten()

# Xuất toàn bộ dữ liệu ra file CSV
output_file_full = r"c:\Users\Admin\Documents\Zalo Received Files\kiemtra\BTL Python\AI\KetQuaDuDoan_Full.xlsx"
results_full.to_csv(output_file_full, index=False)
print(f" Kết quả được lưu vào file: {output_file_full}")



#gd
# Tải mô hình toàn cục
model = None
scaler = None  # Định nghĩa scaler để xử lý dữ liệu đầu vào

# Hàm tải mô hình
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(r"c:\Users\Admin\Documents\Zalo Received Files\kiemtra\BTL Python\AI\model.h5")
        messagebox.showinfo("Thông báo", "Mô hình đã được tải thành công!")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi tải mô hình: {e}")

# Hàm dự đoán
def predict():
    if model is None:
        messagebox.showwarning("Cảnh báo", "Hãy tải mô hình trước khi dự đoán!")
        return

    try:
        # Lấy dữ liệu đầu vào
        credit_score = float(entry_credit_score.get())
        age = float(entry_age.get())
        balance = float(entry_balance.get())
        gender = var_gender.get()
        geography = var_geography.get()

        # Chuyển đổi dữ liệu giống như khi huấn luyện
        input_data = np.array([[credit_score, age, balance]])
        
        # Chuẩn hóa dữ liệu
        if scaler:
            input_data = scaler.transform(input_data)

        # Dự đoán
        prediction = model.predict(input_data)
        result = "Khách hàng có thể rời bỏ" if prediction[0][0] > 0.5 else "Khách hàng không rời bỏ"

        label_result.config(text=result)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {e}")

# Giao diện Tkinter
root = tk.Tk()
root.title("Dự đoán khách hàng rời bỏ ngân hàng")
root.geometry("400x400")

# Nút tải mô hình
tk.Button(root, text="Tải Mô Hình", command=load_model).pack(pady=10)

# Nhập dữ liệu
frame_input = tk.Frame(root)
frame_input.pack(pady=10)

var_gender = tk.StringVar(value="Nam")
var_geography = tk.StringVar(value="France")

tk.Label(frame_input, text="Điểm tín dụng:").grid(row=0, column=0)
entry_credit_score = tk.Entry(frame_input)
entry_credit_score.grid(row=0, column=1)

tk.Label(frame_input, text="Tuổi:").grid(row=1, column=0)
entry_age = tk.Entry(frame_input)
entry_age.grid(row=1, column=1)

tk.Label(frame_input, text="Số dư tài khoản:").grid(row=2, column=0)
entry_balance = tk.Entry(frame_input)
entry_balance.grid(row=2, column=1)

tk.Label(frame_input, text="Giới tính:").grid(row=3, column=0)
tk.OptionMenu(frame_input, var_gender, "Nam", "Nữ").grid(row=3, column=1)

tk.Label(frame_input, text="Quốc gia:").grid(row=4, column=0)
tk.OptionMenu(frame_input, var_geography, "France", "Germany", "Spain").grid(row=4, column=1)

# Nút dự đoán
btn_predict = tk.Button(root, text="Dự đoán", command=predict)
btn_predict.pack(pady=10)

label_result = tk.Label(root, text="")
label_result.pack(pady=10)

root.mainloop()


