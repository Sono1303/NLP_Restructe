
# Natural Language Processing (NLP) - Lab Exercises

Repository này chứa các bài thực hành môn Xử lý Ngôn ngữ Tự nhiên (NLP), bao gồm các kỹ thuật cơ bản đến nâng cao trong lĩnh vực NLP.

## Cấu trúc Repository

```
./
├── src/                # Source code Python
├── spark_labs/         # Source code Scala
├── notebook/           # Jupyter notebooks cho các bài lab
├── test/               # Test scripts và unit tests
├── data/               # Metadata và mô tả datasets (KHÔNG chứa dataset thực tế)
├── report/             # Báo cáo chi tiết cho từng lab
├── README.md           # File này
├── .gitignore          # Loại bỏ files không cần thiết
```

## Danh sách Labs

### Lab 1 & 2: Preprocessing và Vectorization
- **Mô tả**: Xử lý dữ liệu văn bản, tokenization, stemming, lemmatization, CountVectorizer
- **Dataset**: Universal Dependencies English-EWT
- **Báo cáo**: [lab1.md](report/lab1.md)
- **Code**: [src/preprocessing/](src/preprocessing/), [src/representations/](src/representations/)

### Lab 2: Spark MLlib và NLP Pipeline
- **Mô tả**: Sử dụng Apache Spark cho xử lý NLP trên dữ liệu lớn
- **Technologies**: Scala, Spark MLlib
- **Báo cáo**: [lab2.md](report/lab2.md)
- **Code**: [spark_labs/](spark_labs/)

### Lab 3: Word Embeddings
- **Phần 1**: Trực quan hóa và phân tích embedding
  - Báo cáo: [lab3_part1.md](report/lab3_part1.md)
- **Phần 2**: Word2Vec và training embeddings
  - Báo cáo: [lab3_part2.md](report/lab3_part2.md)
- **Datasets**: GloVe, UD English-EWT, C4
- **Code**: [notebook/lab3_word_embeddings.ipynb](notebook/lab3_word_embeddings.ipynb), [test/lab4_*.py](test/)

### Lab 4: Text Classification & Sentiment Analysis
- **Mô tả**: Phân loại văn bản và phân tích cảm xúc với Spark MLlib
- **Models**: Logistic Regression, Naive Bayes, GBT, Neural Networks
- **Báo cáo**: [lab4.md](report/lab4.md)
- **Code**: [src/models/text_classifier.py](src/models/text_classifier.py), [test/lab5_*.py](test/)

### Lab 5: RNN và Deep Learning cho NLP
- **Phần 1**: PyTorch Introduction
  - Báo cáo: [lab5_part1.md](report/lab5_part1.md)
  - Notebook: [lab5_pytorch_introduction.ipynb](notebook/lab5_pytorch_introduction.ipynb)
- **Phần 2**: RNN cho Text Classification
  - Báo cáo: [lab5_part2.md](report/lab5_part2.md)
  - Notebook: [lab5_rnn_text_classification.ipynb](notebook/lab5_rnn_text_classification.ipynb)
- **Phần 3**: RNN cho POS Tagging
  - Báo cáo: [lab5_part3.md](report/lab5_part3.md)
  - Notebook: [lab5_rnn_for_pos_tagging.ipynb](notebook/lab5_rnn_for_pos_tagging.ipynb)
- **Phần 4**: RNN cho Named Entity Recognition (NER)
  - Báo cáo: [lab5_part4.md](report/lab5_part4.md)
  - Notebook: [lab5_rnn_for_ner.ipynb](notebook/lab5_rnn_for_ner.ipynb)
- **Datasets**: HWU-64, UD English-EWT, CoNLL-2003

### Lab 6: Transformers và Attention Mechanism
- **Mô tả**: Giới thiệu về Transformer architecture và Hugging Face
- **Báo cáo**: [lab6.md](report/lab6.md)
- **Notebook**: [lab6_intro_transformers.ipynb](notebook/lab6_intro_transformers.ipynb)

## Technologies & Libraries

- **Python**: PyTorch, Gensim, scikit-learn, pandas, numpy, matplotlib
- **Scala**: Apache Spark MLlib
- **Deep Learning**: RNN, LSTM, Transformers
- **NLP Tools**: Word2Vec, GloVe, TF-IDF

## Datasets

**Lưu ý**: Repository này KHÔNG chứa datasets thực tế để tránh quá tải. Các datasets được mô tả trong [data/README.md](data/README.md).

Các datasets sử dụng:
- Universal Dependencies English-EWT (UD_English-EWT)
- GloVe pre-trained vectors
- C4 dataset (subset)
- HWU-64 (Intent Classification)
- CoNLL-2003 (Named Entity Recognition)
- Sentiment datasets

## Getting Started

### Yêu cầu hệ thống
```bash
# Python 3.8+
pip install -r requirements.txt

# Apache Spark (for Lab 2, 4)
# Scala 2.12+
```

### Chạy các Lab

**Lab 1 (Python)**:
```bash
python test/main.py
python test/lab2_test.py
```

**Lab 2 (Spark/Scala)**:
```bash
cd spark_labs
sbt compile
sbt "runMain com.lhson.spark.Lab17_NLPPipeline"
```

**Lab 3 (Word Embeddings)**:
```bash
python test/lab4_test.py
python test/lab4_embedding_training_demo.py
python test/lab4_spark_word2vec_demo.py
jupyter notebook notebook/lab3_word_embeddings.ipynb
```

**Lab 4 (Text Classification)**:
```bash
python test/lab5_test.py
python test/lab5_spark_sentiment_analysis.py
```

**Lab 5 (RNN/LSTM - PyTorch)**:
```bash
jupyter notebook notebook/lab5_pytorch_introduction.ipynb
jupyter notebook notebook/lab5_rnn_text_classification.ipynb
jupyter notebook notebook/lab5_rnn_for_pos_tagging.ipynb
jupyter notebook notebook/lab5_rnn_for_ner.ipynb
```

**Lab 6 (Transformers)**:
```bash
jupyter notebook notebook/lab6_intro_transformers.ipynb
```

## Báo cáo

Tất cả báo cáo chi tiết cho từng lab được lưu trong thư mục [report/](report/), bao gồm:
- Giải thích các bước thực hiện
- Hướng dẫn chạy code
- Phân tích kết quả và đánh giá
- Khó khăn gặp phải và giải pháp
- Tài liệu tham khảo

## Lưu ý quan trọng

1. **Datasets**: Không push datasets lớn lên GitHub. Sử dụng `.gitignore` để loại bỏ.
2. **Models**: Model files (`.pt`, `.pth`) được ignore, chỉ giữ lại trained models quan trọng.
3. **Reports**: Ưu tiên format markdown (.md) để dễ đọc và trích xuất thông tin.
4. **Code organization**: Code được tổ chức theo từng lab, với structure rõ ràng.

## Tài liệu tham khảo

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Apache Spark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)
- [Universal Dependencies](https://universaldependencies.org/)
- [GloVe: Global Vectors](https://nlp.stanford.edu/projects/glove/)
- [CoNLL-2003 NER Dataset](https://www.clips.uantwerpen.be/conll2003/ner/)

## Cập nhật

- Tuần 12/2025: Tái cấu trúc repository theo chuẩn, hoàn thiện báo cáo Lab 1-6
- Tuần 11/2025: Hoàn thành Lab 5 (RNN/LSTM)
- Tuần 10/2025: Lab 4 (Text Classification & Sentiment Analysis)

---
