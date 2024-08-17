import re
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import os
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def setup_openai_api():
    file_path = r'C:\Users\affc1\OneDrive\Masaüstü\api-key.txt'
    with open(file_path, 'r') as file:
        openai.api_key = file.read().strip()

setup_openai_api()

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

#GPT-4 modelini kullanarak yanıt üretme
def generate_response_with_gpt4(prompt, user_language):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant fluent in {user_language}."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message['content']

# Log dosyasını işleyip, veri listesi oluşturma
def process_logs(log_file_path):
    log_pattern = re.compile(
        r'(?P<ip>\S+) - - \[(?P<time>.*?)] "(?P<request>.*?)" (?P<status>\d+) (?P<size>\d+|-) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
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

# Log verilerini vektörlere dönüştürme ve TF-IDF vektörleştirme
def vectorize_requests(log_data_list):
    requests_data = [
        f"{log['time']} {log['request']} {log['ip']} {log['status']} {log['referer']} {log['user_agent']}"
        for log in log_data_list
    ]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(requests_data)
    return X, vectorizer

# FAISS index oluşturma
def build_faiss_index(X):
    X_array = X.toarray().astype('float32')
    faiss.normalize_L2(X_array)  # L2 normalizasyonu yaparak özellik vektörlerinin uzunluğunu 1 yap
    dimension = X_array.shape[1]  # Vektör boyutunu (özellik sayısını) al
    index = faiss.IndexFlatL2(dimension)  # L2 mesafesine göre düz bir index oluştur
    index.add(X_array)
    return index

# Büyük veriler için k değerini optimize ediyoruz.
def determine_k_value(log_data_list, base_k=5, factor=0.001):
    log_count = len(log_data_list)
    dynamic_k = min(int(base_k + factor * log_count), log_count)
    return dynamic_k

# Kullanıcı sorgusuna göre log verilerinden yanıt üretme
def retrieve_and_generate_response(user_query, vectorizer, faiss_index, log_data_list):
    user_language = detect_language(user_query)
    query_vector = vectorizer.transform([user_query]).toarray().astype('float32')  # Sorguyu vektöre dönüştür
    faiss.normalize_L2(query_vector)  # Sorgu vektörünü normalleştir
    D, I = faiss_index.search(query_vector, k=5)  # Sorguya en yakın 5 log kaydını bul
    similar_logs = [log_data_list[idx] for idx in I[0]]  # Benzer logları listele
    log_texts = "\n".join([f"Request: {log['time']}, {log['request']}, Status: {log['status']}, IP: {log['ip']}" for log in similar_logs])  # Log
    # lardan bilgi stringleri oluştur
    prompt = f"The following are log entries:\n{log_texts}\n\nBased on these logs, {user_query}."  # GPT-4 için prompt hazırla
    response = generate_response_with_gpt4(prompt, user_language)  # GPT-4 kullanarak yanıt üret
    return response

def main():
    setup_openai_api()
    log_file_path = 'C:\\xampp\\apache\\logs\\access.log'
    if not os.path.exists(log_file_path):
        print("Log dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return
    log_data_list = process_logs(log_file_path)
    X, vectorizer = vectorize_requests(log_data_list)
    faiss_index = build_faiss_index(X)

    while True:
        user_query = input("Sorunuzu girin (Çıkmak için 'exit' yazın): ")
        if user_query.lower() == 'exit':
            print("Çıkış yapılıyor...")
            break
        elif len(user_query.strip()) == 0:
            print("Geçersiz soru. Lütfen yeniden deneyin.")
            continue
        response = retrieve_and_generate_response(user_query, vectorizer, faiss_index, log_data_list)
        print(response)

if __name__ == '__main__':
    main()
