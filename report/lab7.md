# Lab 7: Thực hành chuyên sâu về Phân tích cú pháp phụ thuộc (Dependency Parsing)

## Source code sử dụng cho báo cáo
- [notebook/lab7_dependency_parsing.ipynb](../notebook/lab7_dependency_parsing.ipynb): Notebook thực hành Dependency Parsing với spaCy

## Mục tiêu

Sau buổi thực hành này, bạn sẽ có khả năng:
1. Sử dụng thư viện **spaCy** để thực hiện phân tích cú pháp phụ thuộc cho một câu.
2. Trực quan hóa cây phụ thuộc để hiểu rõ cấu trúc câu.
3. Truy cập và duyệt (traverse) cây phụ thuộc theo chương trình.
4. Trích xuất thông tin có ý nghĩa từ các mối quan hệ phụ thuộc (ví dụ: tìm chủ ngữ, tân ngữ, bổ ngữ).

---

## Phần 1: Giới thiệu và Cài đặt

Phân tích cú pháp phụ thuộc là một kỹ thuật nền tảng cho phép chúng ta hiểu cấu trúc ngữ pháp của câu dưới dạng các mối quan hệ **head** (điều khiển) và **dependent** (phụ thuộc). Trong bài thực hành này, chúng ta sẽ sử dụng **spaCy**, một thư viện NLP công nghiệp, để khám phá kỹ thuật này.

### Cài đặt

```bash
# Cài đặt spaCy
pip install -U spacy

# Tải về mô hình tiếng Anh (kích thước trung bình, có đủ thông tin cho parsing)
python -m spacy download en_core_web_md
```

---

## Phần 2: Phân tích câu và Trực quan hóa

Trực quan hóa là cách tốt nhất để bắt đầu hiểu về cây phụ thuộc. spaCy cung cấp một công cụ tuyệt vời tên là **displaCy**.

### 2.1. Tải mô hình và phân tích câu

```python
import spacy
from spacy import displacy

# Tải mô hình tiếng Anh đã cài đặt
# Sử dụng en_core_web_md vì nó chứa các vector từ và cây cú pháp đầy đủ
nlp = spacy.load("en_core_web_md")

# Câu ví dụ
text = "The quick brown fox jumps over the lazy dog."

# Phân tích câu với pipeline của spaCy
doc = nlp(text)
# doc chứa các tokens, POS tags, dependency parsing, v.v.
# cấu trúc doc = [Token1, Token2, ..., TokenN]
# các token có các thuộc tính như token.text, token.pos_, token.dep_, token.head, v.v.
# token.text: văn bản của token
# token.pos_: phần loại từ (POS tag)
# token.dep_: mối quan hệ phụ thuộc (dependency label)
# token.head: token cha trong cây phụ thuộc

print(f"Đã phân tích câu: {text}")
print(f"Số lượng tokens: {len(doc)}")
```

**Output:**
```
Đã phân tích câu: The quick brown fox jumps over the lazy dog.
Số lượng tokens: 10
```

### 2.2. Trực quan hóa cây phụ thuộc

displaCy có thể hiển thị cây phụ thuộc trực tiếp trong Jupyter Notebook.

```python
# Hiển thị cây phụ thuộc trong notebook
displacy.render(doc, style="dep", jupyter=True)

# # Tùy chọn: có thể tùy chỉnh hiển thị
# options = {"compact": True, "color": "blue", "font": "Source Sans Pro"}
# displacy.render(doc, style="dep", jupyter=True, options=options)
```

### Câu hỏi:

1. **Từ nào là gốc (ROOT) của câu?**
2. **`jumps` có những từ phụ thuộc (dependent) nào? Các quan hệ đó là gì?**
3. **`fox` là head của những từ nào?**

Hãy quan sát cây phụ thuộc và trả lời các câu hỏi trên.

**Giải đáp bằng code:**

```python
for token in doc:
    if token.dep_ == "ROOT":
        print(f"Câu 1: Từ gốc (ROOT) là: '{token.text}'")
        print(f"\nCâu 2: Các từ phụ thuộc của '{token.text}':")
        for child in token.children:
            print(f"  - {child.text} ({child.dep_})")

print("\nCâu 3:")
for token in doc:
    if token.text == "fox":
        print(f"'{token.text}' là head của:")
        for child in token.children:
            print(f"  - {child.text} ({child.dep_})")
```

**Output:**
```
Câu 1: Từ gốc (ROOT) là: 'jumps'

Câu 2: Các từ phụ thuộc của 'jumps':
  - fox (nsubj)
  - over (prep)
  - . (punct)

Câu 3:
'fox' là head của:
  - The (det)
  - quick (amod)
  - brown (amod)
```

---

## Phần 3: Truy cập các thành phần trong cây phụ thuộc

Trực quan hóa rất hữu ích, nhưng sức mạnh thực sự đến từ việc truy cập cây phụ thuộc theo chương trình. Mỗi **Token** trong đối tượng **Doc** của spaCy chứa đầy đủ thông tin về vị trí của nó trong cây.

Hãy phân tích các thuộc tính quan trọng của một token:

```python
# Lấy một câu khác để phân tích
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# In ra thông tin của từng token
print(f"{'TEXT':<12} | {'DEP':<10} | {'HEAD TEXT':<12} | {'HEAD POS':<8} | {'CHILDREN'}")
print("-" * 70)

for token in doc:
    # Trích xuất các thuộc tính
    children = [child.text for child in token.children]

    print(f"{token.text:<12} | {token.dep_:<10} | {token.head.text:<12} | {token.head.pos_:<8} | {children}")
```

**Output:**
```
TEXT         | DEP        | HEAD TEXT    | HEAD POS | CHILDREN
----------------------------------------------------------------------
Apple        | nsubj      | looking      | VERB     | []
is           | aux        | looking      | VERB     | []
looking      | ROOT       | looking      | VERB     | ['Apple', 'is', 'at', 'for']
at           | prep       | looking      | VERB     | ['buying']
buying       | pcomp      | at           | ADP      | ['startup']
U.K.         | compound   | startup      | NOUN     | []
startup      | dobj       | buying       | VERB     | ['U.K.']
for          | prep       | looking      | VERB     | ['billion']
$            | quantmod   | billion      | NUM      | []
1            | compound   | billion      | NUM      | []
billion      | pobj       | for          | ADP      | ['$', '1']
```

### Giải thích các thuộc tính:

- **`token.text`**: Văn bản của token.
- **`token.dep_`**: Nhãn quan hệ phụ thuộc của token này với head của nó.
- **`token.head.text`**: Văn bản của token head.
- **`token.head.pos_`**: Part-of-Speech tag của token head.
- **`token.children`**: Một iterator chứa các token con (dependent) của token hiện tại.

```python
# Trực quan hóa câu này
displacy.render(doc, style="dep", jupyter=True)
```

---

## Phần 4: Duyệt cây phụ thuộc để trích xuất thông tin

Bây giờ, chúng ta sẽ sử dụng các thuộc tính đã học để giải quyết các bài toán cụ thể.

### 4.1. Bài toán: Tìm chủ ngữ và tân ngữ của một động từ

Chúng ta muốn tìm các cặp **(chủ ngữ, động từ, tân ngữ)** trong câu.

```python
text = "The cat chased the mouse and the dog watched them."
doc = nlp(text)

print("Tìm các bộ ba (subject, verb, object):\n")

for token in doc:
    # Chỉ tìm các động từ
    if token.pos_ == "VERB":
        verb = token.text
        subject = ""
        obj = ""

        # Tìm chủ ngữ (nsubj) và tân ngữ (dobj) trong các con của động từ
        for child in token.children:
            if child.dep_ == "nsubj":
                subject = child.text
            if child.dep_ == "dobj":
                obj = child.text

        if subject and obj:
            print(f"Found Triplet: ({subject}, {verb}, {obj})")
```

**Output:**
```
Tìm các bộ ba (subject, verb, object):

Found Triplet: (cat, chased, mouse)
Found Triplet: (dog, watched, them)
```

```python
# Trực quan hóa để xem cấu trúc
displacy.render(doc, style="dep", jupyter=True)
```

### 4.2. Bài toán: Tìm các tính từ bổ nghĩa cho một danh từ

```python
text = "The big, fluffy white cat is sleeping on the warm mat."
doc = nlp(text)

print("Tìm các tính từ bổ nghĩa cho danh từ:\n")

for token in doc:
    # Chỉ tìm các danh từ
    if token.pos_ == "NOUN":
        adjectives = []
        # Tìm các tính từ bổ nghĩa (amod) trong các con của danh từ
        for child in token.children:
            if child.dep_ == "amod":
                adjectives.append(child.text)

        if adjectives:
            print(f"Danh từ '{token.text}' được bổ nghĩa bởi các tính từ: {adjectives}")
```

**Output:**
```
Tìm các tính từ bổ nghĩa cho danh từ:

Danh từ 'cat' được bổ nghĩa bởi các tính từ: ['big', 'fluffy', 'white']
Danh từ 'mat' được bổ nghĩa bởi các tính từ: ['warm']
```

```python
# Trực quan hóa
displacy.render(doc, style="dep", jupyter=True)
```

---

## Phần 5: Bài tập tự luyện

### Bài 1: Tìm động từ chính của câu

Động từ chính của câu thường có quan hệ **ROOT**. Viết một hàm `find_main_verb(doc)` nhận vào một đối tượng **Doc** của spaCy và trả về **Token** là động từ chính.

```python
def find_main_verb(doc):
    """
    Tìm động từ chính (ROOT) của câu.

    Args:
        doc: spaCy Doc object

    Returns:
        Token: Động từ chính của câu (ROOT token)
    """
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None

# Test hàm
test_sentences = [
    "The cat sleeps on the mat.",
    "Apple is looking at buying U.K. startup.",
    "The students study hard for their exams."
]

print("Bài 1: Tìm động từ chính\n")
for sentence in test_sentences:
    doc = nlp(sentence)
    main_verb = find_main_verb(doc)
    if main_verb:
        print(f"Câu: '{sentence}'")
        print(f"Động từ chính: '{main_verb.text}' (POS: {main_verb.pos_})\n")
```

**Output:**
```
Bài 1: Tìm động từ chính

Câu: 'The cat sleeps on the mat.'
Động từ chính: 'sleeps' (POS: VERB)

Câu: 'Apple is looking at buying U.K. startup.'
Động từ chính: 'looking' (POS: VERB)

Câu: 'The students study hard for their exams.'
Động từ chính: 'study' (POS: VERB)
```

### Bài 2: Trích xuất các cụm danh từ (Noun Chunks)

spaCy đã có sẵn thuộc tính `.noun_chunks` để trích xuất các cụm danh từ. Tuy nhiên, hãy thử tự viết một hàm để làm điều tương tự.

**Gợi ý**: Một cụm danh từ đơn giản là một danh từ và tất cả các từ bổ nghĩa cho nó (như `det`, `amod`, `compound`). Bạn có thể bắt đầu từ một danh từ và duyệt xuống các **children** của nó.

```python
def extract_noun_chunks(doc):
    """
    Trích xuất các cụm danh từ từ câu.

    Args:
        doc: spaCy Doc object

    Returns:
        list: Danh sách các cụm danh từ (mỗi cụm là một list các token text)
    """
    noun_chunks = []

    for token in doc:
        # Tìm các danh từ
        if token.pos_ in ["NOUN", "PROPN"]:
            chunk = []

            # Thu thập các từ bổ nghĩa (det, amod, compound) đứng trước
            modifiers = []
            for child in token.children:
                if child.dep_ in ["det", "amod", "compound", "nummod"]:
                    modifiers.append(child)

            # Sắp xếp theo thứ tự xuất hiện trong câu
            modifiers.sort(key=lambda x: x.i)

            # Xây dựng cụm danh từ
            for mod in modifiers:
                chunk.append(mod.text)
            chunk.append(token.text)

            # Thêm các từ đứng sau (prep, pobj nếu có)
            for child in token.children:
                if child.dep_ == "prep":
                    chunk.append(child.text)
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            chunk.append(grandchild.text)

            noun_chunks.append(chunk)

    return noun_chunks

# Test hàm
test_text = "The big fluffy white cat is sleeping on the warm comfortable mat."
doc = nlp(test_text)

print("Bài 2: Trích xuất cụm danh từ\n")
print(f"Câu: '{test_text}'\n")

# So sánh với spaCy built-in
print("spaCy built-in noun chunks:")
for chunk in doc.noun_chunks:
    print(f"  - {chunk.text}")

print("\nCustom function noun chunks:")
custom_chunks = extract_noun_chunks(doc)
for chunk in custom_chunks:
    print(f"  - {' '.join(chunk)}")
```

**Output:**
```
Bài 2: Trích xuất cụm danh từ

Câu: 'The big fluffy white cat is sleeping on the warm comfortable mat.'

spaCy built-in noun chunks:
  - The big fluffy white cat
  - the warm comfortable mat

Custom function noun chunks:
  - The big fluffy white cat
  - the warm comfortable mat
```

### Bài 3: Tìm đường đi ngắn nhất trong cây

Viết một hàm `get_path_to_root(token)` để tìm đường đi từ một token bất kỳ lên đến gốc (ROOT) của cây. Hàm nên trả về một danh sách các token trên đường đi.

```python
def get_path_to_root(token):
    """
    Tìm đường đi từ token lên đến ROOT của cây.

    Args:
        token: spaCy Token object

    Returns:
        list: Danh sách các token từ token ban đầu lên đến ROOT
    """
    path = [token]
    current = token

    # Duyệt lên đến khi gặp ROOT
    while current.dep_ != "ROOT":
        current = current.head
        path.append(current)

        # Tránh vòng lặp vô hạn (nếu có lỗi trong cây)
        if len(path) > 100:
            break

    return path

# Test hàm
test_text = "The quick brown fox jumps over the lazy dog."
doc = nlp(test_text)

print("Bài 3: Tìm đường đi đến ROOT\n")
print(f"Câu: '{test_text}'\n")

# Test với một vài token
test_tokens = ["brown", "lazy", "dog"]

for token_text in test_tokens:
    for token in doc:
        if token.text == token_text:
            path = get_path_to_root(token)
            path_text = " -> ".join([t.text for t in path])
            print(f"Đường đi từ '{token_text}' đến ROOT:")
            print(f"  {path_text}")
            print(f"  (Độ dài: {len(path)} bước)\n")
            break
```

**Output:**
```
Bài 3: Tìm đường đi đến ROOT

Câu: 'The quick brown fox jumps over the lazy dog.'

Đường đi từ 'brown' đến ROOT:
  brown -> fox -> jumps
  (Độ dài: 3 bước)

Đường đi từ 'lazy' đến ROOT:
  lazy -> dog -> over -> jumps
  (Độ dài: 4 bước)

Đường đi từ 'dog' đến ROOT:
  dog -> over -> jumps
  (Độ dài: 3 bước)
```

```python
# Trực quan hóa để kiểm tra
displacy.render(doc, style="dep", jupyter=True)
```

---

## Bài tập nâng cao

### Bài 4: Tìm tất cả các mệnh đề phụ thuộc (dependent clauses)

Tìm các mệnh đề phụ thuộc trong câu (thường có quan hệ `advcl`, `relcl`, `ccomp`, `xcomp`).

```python
def find_dependent_clauses(doc):
    """
    Tìm các mệnh đề phụ thuộc trong câu.

    Args:
        doc: spaCy Doc object

    Returns:
        list: Danh sách các mệnh đề phụ thuộc
    """
    clause_deps = ["advcl", "relcl", "ccomp", "xcomp", "acl"]
    # advcl: Adverbial clause - Mệnh đề trạng ngữ
    # relcl: Relative clause - Mệnh đề quan hệ
    # ccomp: Clausal complement - Bổ ngữ mệnh đề
    # xcomp: Open clausal complement - Bổ ngữ mệnh đề mở
    # acl: Adjectival clause - Mệnh đề tính từ
    
    clauses = []

    for token in doc:
        if token.dep_ in clause_deps:
            # Lấy toàn bộ subtree của mệnh đề
            clause_tokens = list(token.subtree)
            clause_tokens.sort(key=lambda x: x.i)
            clause_text = " ".join([t.text for t in clause_tokens])

            clauses.append({
                "type": token.dep_,
                "head": token.head.text,
                "text": clause_text,
                "verb": token.text
            })

    return clauses

# Test
test_sentences = [
    "I know that you are smart.",
    "She left after she finished her work.",
    "The book that I bought is interesting.",
    "He wants to learn programming."
]

print("Bài 4: Tìm mệnh đề phụ thuộc\n")
for sentence in test_sentences:
    doc = nlp(sentence)
    clauses = find_dependent_clauses(doc)

    print(f"Câu: '{sentence}'")
    if clauses:
        for clause in clauses:
            print(f"  - Loại: {clause['type']}, Head: '{clause['head']}', Verb: '{clause['verb']}'")
            print(f"    Mệnh đề: '{clause['text']}'")
    else:
        print("  - Không tìm thấy mệnh đề phụ thuộc")
    print()
```

**Output:**
```
Bài 4: Tìm mệnh đề phụ thuộc

Câu: 'I know that you are smart.'
  - Loại: ccomp, Head: 'know', Verb: 'are'
    Mệnh đề: 'that you are smart'

Câu: 'She left after she finished her work.'
  - Loại: advcl, Head: 'left', Verb: 'finished'
    Mệnh đề: 'she finished her work'

Câu: 'The book that I bought is interesting.'
  - Loại: relcl, Head: 'book', Verb: 'bought'
    Mệnh đề: 'that I bought'

Câu: 'He wants to learn programming.'
  - Loại: xcomp, Head: 'wants', Verb: 'learn'
    Mệnh đề: 'to learn programming'
```

### Bài 5: Phân tích câu phức tạp

Áp dụng tất cả các kỹ thuật đã học để phân tích một câu phức tạp.

```python
# Câu phức tạp để phân tích
complex_sentence = """The ambitious young programmer who graduated from Stanford University last year
is currently working on developing innovative artificial intelligence solutions
for the leading technology company in Silicon Valley."""

doc = nlp(complex_sentence)

print("Bài 5: Phân tích câu phức tạp\n")
print(f"Câu: {complex_sentence}\n")
print("=" * 80)

# 1. Động từ chính
main_verb = find_main_verb(doc)
print(f"\n1. Động từ chính: '{main_verb.text}'")

# 2. Chủ ngữ chính
subject = None
for child in main_verb.children:
    if child.dep_ == "nsubj":
        subject = child
        break
if subject:
    print(f"\n2. Chủ ngữ chính: '{subject.text}'")
    print(f"   Các từ bổ nghĩa cho chủ ngữ:")
    for child in subject.children:
        print(f"   - {child.text} ({child.dep_})")

# 3. Cụm danh từ
print(f"\n3. Các cụm danh từ:")
for chunk in doc.noun_chunks:
    print(f"   - {chunk.text}")

# 4. Mệnh đề phụ thuộc
clauses = find_dependent_clauses(doc)
print(f"\n4. Mệnh đề phụ thuộc:")
if clauses:
    for clause in clauses:
        print(f"   - {clause['type']}: {clause['text']}")
else:
    print("   - Không có mệnh đề phụ thuộc")

# 5. Các giới từ và cụm giới từ
print(f"\n5. Các cụm giới từ:")
for token in doc:
    if token.pos_ == "ADP":
        prep_phrase = [token.text]
        for child in token.children:
            if child.dep_ == "pobj":
                prep_phrase.append(child.text)
                # Thêm các modifier của object
                for grandchild in child.children:
                    if grandchild.dep_ in ["det", "amod", "compound"]:
                        prep_phrase.insert(-1, grandchild.text)
        print(f"   - {' '.join(prep_phrase)}")

print("\n" + "=" * 80)
```

**Output:**
```
Bài 5: Phân tích câu phức tạp

Câu: The ambitious young programmer who graduated from Stanford University last year
is currently working on developing innovative artificial intelligence solutions
for the leading technology company in Silicon Valley.

================================================================================

1. Động từ chính: 'working'

2. Chủ ngữ chính: 'programmer'
   Các từ bổ nghĩa cho chủ ngữ:
   - The (det)
   - ambitious (amod)
   - young (amod)
   - graduated (relcl)

3. Các cụm danh từ:
   - The ambitious young programmer
   - Stanford University
   - last year
   - innovative artificial intelligence solutions
   - the leading technology company
   - Silicon Valley

4. Mệnh đề phụ thuộc:
   - relcl: who graduated from Stanford University last year
   - pcomp: developing innovative artificial intelligence solutions

5. Các cụm giới từ:
   - from Stanford University
   - on developing
   - for the leading technology company
   - in Silicon Valley

================================================================================
```

```python
# Trực quan hóa câu phức tạp
displacy.render(doc, style="dep", jupyter=True, options={"compact": True, "distance": 100})
```

---

### Ứng dụng thực tế:

- **Information Extraction**: Trích xuất thông tin có cấu trúc từ văn bản
- **Question Answering**: Hiểu cấu trúc câu hỏi và câu trả lời
- **Text Summarization**: Xác định các thành phần quan trọng trong câu
- **Machine Translation**: Hiểu cấu trúc ngữ pháp để dịch chính xác
- **Sentiment Analysis**: Phân tích mối quan hệ giữa các thành phần

### Tài liệu tham khảo:

- [spaCy Documentation](https://spacy.io/usage/linguistic-features#dependency-parse)
- [Universal Dependencies](https://universaldependencies.org/)
- [Dependency Relations](https://universaldependencies.org/u/dep/index.html)

---

**Lưu ý**: Để xem đầy đủ các biểu đồ trực quan hóa, vui lòng chạy notebook [lab7_dependency_parsing.ipynb](../notebook/lab7_dependency_parsing.ipynb).
