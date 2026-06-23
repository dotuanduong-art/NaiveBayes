import math
import pandas as pd
from sklearn.model_selection import train_test_split


def split_text(text):
    return text.lower().split()


def train(dataset):
    word_counts = {}
    class_counts = {}
    vocab = set()

    for row in dataset:
        label = row.label
        text = row.text

        if label not in word_counts:
            word_counts[label] = {}
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
        words = split_text(text)
        for word in words:
            vocab.add(word)
            if word not in word_counts[label]:
                word_counts[label][word] = 0
            word_counts[label][word] += 1

    return word_counts, class_counts, vocab


def predict(text, word_counts, class_counts, vocab):
    words = split_text(text)
    total_docs = sum(class_counts.values())
    best_label = None
    best_score = None

    for label in class_counts:
        # Log xác suất tiên nghiệm P(label) 
        score = math.log(class_counts[label] / total_docs)

        # Tổng số từ trong lớp label
        total_words = sum(word_counts[label].values())

        # Log xác suất có điều kiện P(word|label) với Laplace smoothing
        for word in words:
            word_freq = word_counts[label].get(word,0)
            score += math.log((word_freq + 1) / (total_words + len(vocab)))

        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    return best_label


def doc_file_csv(filename):
    df = pd.read_csv(filename, encoding="latin1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    return list(df.itertuples(index=False))


def main():
    # Bước 1: Đọc dữ liệu
    dataset = doc_file_csv("spam.csv")
    print(f"Tong so tin nhan: {len(dataset)}")

    # Bước 2: Chia train/test (80/20)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"So luong train: {len(train_data)}")
    print(f"So luong test : {len(test_data)}")

    # Bước 3: Huấn luyện mô hình
    word_counts, class_counts, vocab = train(train_data)
    print(f"Kich thuoc tu vung: {len(vocab)} tu")

    # Bước 4: Dự đoán và đánh giá
    spam_messages = []
    ham_messages = []
    correct = 0

    for row in test_data:
        predicted = predict(row.text, word_counts, class_counts, vocab)
        if predicted == row.label:
            correct += 1
        if predicted == "spam":
            spam_messages.append((row.text, row.label, predicted))
        else:
            ham_messages.append((row.text, row.label, predicted))

    # Bước 5: In kết quả
    accuracy = correct / len(test_data) * 100
    print("\n" + "="*60)
    print(f"So tin nhan du doan la SPAM : {len(spam_messages)}")
    print(f"So tin nhan du doan la HAM  : {len(ham_messages)}")
    print(f"So du doan dung             : {correct}/{len(test_data)}")
    print(f"Do chinh xac (Accuracy)     : {accuracy:.2f}%")
    print("="*60)


    #Thử dự đoán tin mới
    while True:
        message= input("Nhập tin cần kiểm tra : ")
        if message=="":
            break
        result= predict(message,word_counts,class_counts,vocab)
        if result=="spam": 
            print("Thư rác (SPAM)")
        else:
            print("Đây là thư hợp lệ (HAM)")

if __name__ == "__main__":
    main()
