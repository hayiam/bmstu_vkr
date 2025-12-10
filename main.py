import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Определитель спама")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Загрузка модели и векторизатора
        try:
            with open('svc_model.pkl', 'rb') as model_file:
                self.model = pickle.load(model_file)
            
            with open('vectorizer.pkl', 'rb') as vectorizer_file:
                self.vectorizer = pickle.load(vectorizer_file)
            
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")
            self.model_loaded = False
        
        self.create_widgets()
    
    def create_widgets(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(main_frame, 
                               text="Определитель спам-сообщений", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Метка и поле ввода
        input_label = ttk.Label(main_frame, text="Введите сообщение для проверки:")
        input_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Текстовое поле с прокруткой
        self.text_area = scrolledtext.ScrolledText(main_frame, 
                                                  height=8, 
                                                  width=70, 
                                                  font=("Arial", 10))
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.text_area.focus()
        
        # Фрейм для кнопок
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Кнопка проверки
        self.check_button = ttk.Button(button_frame, 
                                      text="Проверить на спам", 
                                      command=self.check_spam,
                                      state=tk.NORMAL if self.model_loaded else tk.DISABLED)
        self.check_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Кнопка очистки
        clear_button = ttk.Button(button_frame, 
                                 text="Очистить", 
                                 command=self.clear_text)
        clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Кнопка выхода
        exit_button = ttk.Button(button_frame, 
                                text="Выход", 
                                command=self.root.quit)
        exit_button.pack(side=tk.LEFT)
        
        # Область для результата
        result_frame = ttk.LabelFrame(main_frame, text="Результат", padding="10")
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = ttk.Label(result_frame, 
                                     text="Введите сообщение и нажмите 'Проверить на спам'", 
                                     font=("Arial", 11))
        self.result_label.pack(fill=tk.X)
        
        # Статус бар
        self.status_bar = ttk.Label(main_frame, 
                                   text="Готов к работе" if self.model_loaded else "Ошибка загрузки модели", 
                                   relief=tk.SUNKEN, 
                                   anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
    
    def check_spam(self):
        if not self.model_loaded:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return
        
        message = self.text_area.get(1.0, tk.END).strip()
        
        if not message:
            messagebox.showwarning("Предупреждение", "Пожалуйста, введите сообщение для проверки.")
            return
        
        try:
            # Преобразование входного сообщения в вектор
            message_vectorized = self.vectorizer.transform([message])
            # Прогнозирование с помощью модели
            prediction = self.model.predict(message_vectorized)
            result = prediction[0]
            
            # Обновление интерфейса в зависимости от результата
            if result == 1:
                self.result_label.config(text="⚠️  Это сообщение является СПАМОМ!", foreground="red")
            else:
                self.result_label.config(text="✓  Это сообщение НЕ является спамом", foreground="green")
            
            self.status_bar.config(text="Проверка завершена")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при проверке: {str(e)}")
            self.status_bar.config(text="Ошибка при проверке")
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.result_label.config(text="Введите сообщение и нажмите 'Проверить на спам'", foreground="black")
        self.status_bar.config(text="Текст очищен")
        self.text_area.focus()

def main():
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
