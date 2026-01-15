# Vietnamese SMS Spam Detection

Dự án sử dụng học máy để phân loại tin nhắn rác (Spam) và tin nhắn thường (Ham) cho tiếng Việt.

## 🛠 Cài đặt
1. Clone dự án: `git clone <link-cua-ban>`
2. Tạo môi trường ảo: `python -m venv venv`
3. Kích hoạt venv: `.\venv\Scripts\activate`
4. Cài đặt thư viện: `pip install -r requirements.txt`

## 🧪 Quy trình thực hiện
- **Tiền xử lý:** Tách từ tiếng Việt (PyVi), xóa ký tự đặc biệt, chuẩn hóa chữ thường.
- **Trích xuất đặc trưng:** TF-IDF Vectorizer (N-gram 1,2).
- **Mô hình:** Stacking Classifier (kết hợp nhiều mô hình base).

## 📈 Kết quả
Mô hình đạt độ chính xác cao trên tập dữ liệu thử nghiệm.