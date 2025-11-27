# Lab 3: Word Embeddings

## Phần 1: Trực quan hóa và phân tích embedding

### 1. Source code, dữ liệu, kết quả sử dụng
- [notebook/lab3_word_embeddings.ipynb](../notebook/lab3_word_embeddings.ipynb): Notebook chính trực quan hóa embedding

### 2. Các bước thực hiện
1. Load pre-trained GloVe vectors bằng Gensim
2. Giảm chiều vector từ 100D xuống 2D bằng PCA
3. Trực quan hóa các từ trong không gian 2D bằng matplotlib
4. Tìm kiếm Top K từ tương đồng với một từ bất kỳ (ví dụ: "king")
5. Hiển thị kết quả và phân tích cụm từ, độ tương đồng

### 3. Hướng dẫn chạy code
- Mở notebook `Lab3.ipynb` bằng Jupyter hoặc Colab
- Chạy tuần tự các cell để hiển thị kết quả, hình ảnh trực quan hóa
- Đảm bảo file GloVe vectors đã được giải nén vào đúng thư mục

### 4. Hình ảnh trực quan hóa embedding
*Các hình ảnh trực quan hóa được tạo ra trong notebook lab3_word_embeddings.ipynb*


### 5. Nhận xét về độ tương đồng và các từ đồng nghĩa
- Các từ đồng nghĩa/tương đồng tìm được từ model pre-trained GloVe rất hợp lý, ví dụ "computer" → computers, software, technology
- Độ tương đồng cosine cao cho thấy model học tốt các mối quan hệ ngữ nghĩa

### 6. Phân tích biểu đồ trực quan hóa
- Các từ liên quan được nhóm lại gần nhau, ví dụ cụm hoàng gia (king, queen, prince, kingdom)
- Một số cụm thú vị: công nghệ (computer, software, technology), địa lý (country, city, state)
- Giải thích: GloVe học từ thống kê đồng xuất hiện, PCA bảo toàn khoảng cách tương đối

### 7. So sánh model pre-trained và model tự huấn luyện
- Pre-trained GloVe có chất lượng tốt nhất do data lớn, similarity scores cao
- Model tự huấn luyện (Word2Vec) có vocabulary nhỏ hơn, chất lượng vừa phải nhưng vẫn học được các mối quan hệ cơ bản
- Spark MLlib training trên dataset lớn cho kết quả robust hơn, vocabulary lớn

### 8. Khó khăn và giải pháp
- Xử lý file GloVe lớn, cần đủ RAM
- PCA trên nhiều vector tốn thời gian, cần subset cho visualization
- Khác biệt format vector giữa Gensim và Spark, cần chuẩn hóa
- Đã giải quyết bằng cách subset, tối ưu plotting, dùng wrapper class cho embedding

### 9. Nguồn tham khảo
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)