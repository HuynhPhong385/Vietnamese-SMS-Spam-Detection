import numpy as np
import re
from pyvi import ViTokenizer
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.state import TrainingState

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        self.config = Config()

    def preprocess_vietnamese_text(self, text: str) -> str:
        """
        Hàm hỗ trợ làm sạch văn bản Tiếng Việt
        """
        try:
            # 1. Chuyển về chữ thường và ép kiểu string
            text = str(text).lower()
            # 2. Loại bỏ dấu (giúp nhận diện tốt hơn tin nhắn không dấu)
            text = unidecode(text)
            # 3. Xóa ký tự đặc biệt và số
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            # 4. Tách từ bằng PyVi (Ví dụ: "trúng thưởng" -> "trúng_thưởng")
            text = ViTokenizer.tokenize(text)
            return text
        except Exception as e:
            logger.error(f"Lỗi khi xử lý văn bản: {str(e)}")
            return text

    def transform_data(self, state: TrainingState) -> TrainingState:
        logger.info("Bắt đầu quy trình Data Transformation cho Tiếng Việt")
        try:
            data = state.training_data.copy()
            
            # 1. Xử lý giá trị thiếu (NaN)
            data.dropna(inplace=True)
            
            # 2. Encode labels: spam -> 0, ham -> 1 (Khớp với logic Lab của bạn)
            # Giả sử cột nhãn của bạn tên là 'labels'
            data.loc[data['labels'] == 'spam', 'labels'] = 0
            data.loc[data['labels'] == 'ham', 'labels'] = 1
            data['labels'] = data['labels'].astype(int)
            
            logger.info(f"Label encoding xong. Dữ liệu: {data.shape}")
            
            # 3. Tiền xử lý văn bản Tiếng Việt
            logger.info("Đang thực hiện tách từ và xóa dấu Tiếng Việt...")
            # Giả sử cột nội dung của bạn tên là 'texts_vi'
            data['cleaned_text'] = data['texts_vi'].apply(self.preprocess_vietnamese_text)
            
            # 4. Chia features và target
            X = data['cleaned_text']
            y = np.array(data['labels'], dtype=int)
            
            # 5. Chia tập Train/Test (Tỷ lệ 70:30, giữ nguyên phân lớp stratify)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            logger.info(f"Chia tập dữ liệu xong. Train: {len(X_train)}, Test: {len(X_test)}")
            
            # 6. Vector hóa bằng TF-IDF tối ưu cho Tiếng Việt
            # Sử dụng ngram_range=(1, 2) để bắt được các từ ghép như "trúng_thưởng"
            tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), 
                min_df=2, 
                max_features=5000
            )
            
            # QUAN TRỌNG: Fit trên tập Train, chỉ Transform trên tập Test
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)
            
            logger.info(f"TF-IDF hoàn tất. Số lượng đặc trưng (Features): {X_train_tfidf.shape[1]}")
            
            # 7. Lưu kết quả vào state
            state.transformed_data = data
            state.X_train = X_train
            state.X_test = X_test
            state.y_train = y_train
            state.y_test = y_test
            state.X_train_tfidf = X_train_tfidf
            state.X_test_tfidf = X_test_tfidf
            state.tfidf_vectorizer = tfidf_vectorizer
            
            logger.info("Hoàn thành Data Transformation thành công!")
            return state

        except Exception as e:
            logger.error(f"Thất bại trong bước biến đổi dữ liệu: {str(e)}")
            raise e