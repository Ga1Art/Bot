import logging
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut

# Загрузка данных для nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

# База знаний
faq = {
    "как сбросить пароль": "Чтобы сбросить пароль, перейдите на страницу входа и нажмите 'Забыли пароль'. Следуйте инструкциям, отправленным на вашу почту.",
    "как создать нового пользователя": "Для создания нового пользователя перейдите в раздел 'Управление пользователями' и нажмите 'Добавить нового пользователя'. Заполните необходимые данные и нажмите 'Сохранить'.",
    "как создать отчет": "Чтобы создать отчет, перейдите в раздел 'Отчеты', выберите тип отчета и нажмите 'Создать'.",
    "система работает медленно": "Если система работает медленно, проверьте ваше интернет-соединение и закройте ненужные приложения. Если проблема сохраняется, обратитесь в техническую поддержку.",
    # В дальнейшем, на основании частых вопросов пользователей можно дополнить базу знаний и, при желании, вынести в отдельный файл для удобства редактирования
}

# Идентификатор чата консультанта-человека
OPERATOR_CHAT_ID = '-1002233557021'

# Функция для предобработки текста
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('russian'))
    tokens = word_tokenize(text.lower(), language='russian')
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

questions = list(faq.keys())
preprocessed_questions = [preprocess_text(question) for question in questions]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(questions)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(preprocessed_questions)
sequences = tokenizer.texts_to_sequences(preprocessed_questions)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20)

#Обучение модели
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=1000, output_dim=32, input_length=20),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(faq), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Путь для сохранения и загрузки модели
MODEL_PATH = r'C:\Users\galee\New_Bot\faq_model.h5'

# Проверка существования модели и загрузка её, если она существует
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Модель загружена из файла.")
else:
    model = create_model()
    history = model.fit(padded_sequences, encoded_labels, epochs=1000, verbose=1)
    model.save(MODEL_PATH)
    logger.info("Модель обучена и сохранена в файл. Перезапустите программу для продолжения")

    # Визуализация точности и потерь
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('Model Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(['Accuracy', 'Loss'])
    plt.show()

# Настройка порога уверенности
CONFIDENCE_THRESHOLD = 0.7

def get_model_response(user_input):
    preprocessed_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([preprocessed_input])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=20)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    if confidence < CONFIDENCE_THRESHOLD:  # Порог уверенности
        logger.warning(f"Low confidence ({confidence}) for input: {user_input}")
        return None  # Если уверенность ниже порога, возвращаем None
    
    return faq[questions[predicted_label]]

# Обработчики команд
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Received /start command")
    await update.message.reply_text('Привет! Я бот поддержки ERP-системы. Как я могу помочь?')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Received /help command")
    await update.message.reply_text('Вы можете спросить меня о сбросе пароля, создании нового пользователя, создании отчета и проблемах с производительностью.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text
    logger.info(f"Received message: {user_input}")
    response = get_model_response(user_input)
    
    if response:
        try:
            await update.message.reply_text(response)
            logger.info(f"Sent response: {response}")
        except TimedOut:
            logger.error("Timeout error while sending response. Retrying...")
            await update.message.reply_text("Произошла ошибка соединения. Пожалуйста, попробуйте снова.")
    else:
        try:
            await update.message.reply_text(f"К сожалению, я не могу ответить на ваш вопрос. Соединяю вас с оператором. Пожалуйста, перейдите по ссылке: https://t.me/+xKZ9jGDY3r1lMGMy")
            # Отправка сообщения оператору
            operator_message = f"Пользователь {update.effective_chat.full_name} задал вопрос: {user_input}"
            await context.bot.send_message(chat_id=OPERATOR_CHAT_ID, text=operator_message)
            logger.info(f"Sent message to operator: {operator_message}")
        except TimedOut:
            logger.error("Timeout error while sending message to operator. Retrying...")
            await update.message.reply_text("Произошла ошибка соединения. Пожалуйста, попробуйте снова.")
        except Exception as e:
            logger.error(f"Error while sending message to operator: {e}")
            await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте снова позже.")

def main() -> None:
    # Здесь вставим токен созданного бота
    application = ApplicationBuilder().token("7379796532:AAGHN_UxUPD8v5rw-P-ABnowyuWHy2KGwgw").build()
    logger.info("Bot started")

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == "__main__":
    main()