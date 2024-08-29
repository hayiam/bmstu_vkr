import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Загрузка модели и векторизатора из файлов
with open('svc_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def is_spam(message):
    # Преобразование входного сообщения в вектор
    message_vectorized = vectorizer.transform([message])
    # Прогнозирование с помощью модели
    prediction = model.predict(message_vectorized)
    return prediction[0]

def main():
    print("Добро пожаловать в приложение для определения спама!")
    while True:
        # Запрос сообщения у пользователя
        user_input = input("Введите сообщение (или 'exit' для выхода): ")
        if user_input.lower() == 'exit':
            print("Выход из приложения.")
            break
        
        # Оценка сообщения
        result = is_spam(user_input)
        if result == 1:
            print("Это сообщение является спамом.")
        else:
            print("Это сообщение не является спамом.")

if __name__ == "__main__":
    main()
