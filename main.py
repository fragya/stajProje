import re
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def process_logs(log_file_path):
    log_pattern = re.compile(
        r'(?P<ip>\S+) - - \[(?P<time>.*?)\] "(?P<request>.*?)" (?P<status>\d+) (?P<size>\d+|-) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
    )
    log_data_list = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = log_pattern.match(line)
            if match:
                log_data = match.groupdict()
                if log_data['size'] == '-':
                    log_data['size'] = '0'
                log_data_list.append(log_data)
    return log_data_list

def vectorize_requests(log_data_list):
    requests_data = [
        f"{log['request']} {log['ip']} {log['status']} {log['referer']} {log['user_agent']}"
        for log in log_data_list
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(requests_data)
    return X, vectorizer

def build_faiss_index(X):
    X_array = X.toarray().astype('float32')
    faiss.normalize_L2(X_array)
    dimension = X_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(X_array)
    return index



def main():
    log_file_path = 'C:\\xampp\\apache\\logs\\access.log'
    if not os.path.exists(log_file_path):
        print("Log dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return
    log_data_list = process_logs(log_file_path)
    X, vectorizer = vectorize_requests(log_data_list)
    faiss_index = build_faiss_index(X)


if __name__ == '__main__':
    main()
