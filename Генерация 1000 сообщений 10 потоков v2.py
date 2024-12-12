import asyncio
from langchain.schema import HumanMessage
from langchain_community.chat_models.gigachat import GigaChat
import pandas as pd
import time
import re
import pymorphy3
from tqdm.asyncio import tqdm_asyncio  # Для асинхронного tqdm

# Аутентификационный токен
auth = '...'

# Инициализация клиента GigaChat
chat_client = GigaChat(
    credentials=auth,
    model='GigaChat-Pro',
    scope="GIGACHAT_API_CORP",
    auth_url="https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth",
    max_tokens=3000,
    temperature=0.8,
    verify_ssl_certs=False
)

# Промпт для генерации SMS-сообщений
prompt_template = """
Сгенерируй смс-сообщение для клиента. Напиши три или четыре предложения.
Описание предложения: Необходимо предложить клиенту оформить дебетовую премиальную бизнес-карту Mastercard Preferred. Обслуживание карты стоит 700 рублей в месяц, но клиент может пользоваться ей бесплатно. Что необходимо сделать, чтобы воспользоваться предложением:
1. Оформить премиальную бизнес-карту в офисе банка или онлайн в интернет-банке СберБизнес.
2. Забрать карту.
3. В течение календарного месяца совершить по ней покупки на сумму от 100 000 рублей.
4. В течение следующего месяца пользоваться ей бесплатно.
Преимущества: Предложение по бесплатному обслуживанию — бессрочное.
Оплата покупок без отчётов и платёжных поручений.
Платёжные документы без комиссии.
Лимиты на расходы сотрудников.
Мгновенные переводы на карты любых банков.
В тексте смс запрещено использование:
- Запрещенные слова: № один, номер один, № 1, вкусный, дешёвый, продукт, спам, банкротство, долги, займ, срочно, лучший, главный, номер 1, гарантия, успех, лидер;
- Обращение к клиенту;
- Приветствие клиента;
- Обещания и гарантии;
- Использовать составные конструкции из двух глаголов;
- Причастия и причастные обороты;
- Деепричастия и деепричастные обороты;
- Превосходная степень прилагательных;
- Страдательный залог;
- Порядковые числительные от 10 прописью;
- Цепочки с придаточными предложениями;
- Разделительные повторяющиеся союзы;
- Вводные конструкции;
- Усилители;
- Паразиты времени;
- Несколько существительных подряд, в том числе отглагольных;
- Производные предлоги;
- Сложные предложения, в которых нет связи между частями;
- Сложноподчинённые предложения;
- Даты прописью;
- Близкие по смыслу однородные члены предложения;
- Шокирующие, экстравагантные, кликбейтные фразы;
- Абстрактные заявления без поддержки фактами и отсутствие доказательства пользы для клиента;
- Гарантирующие фразы;
- Узкоспециализированные термины;
- Фразы, способные создать двойственное ощущение, обидеть;
- Речевые клише, рекламные штампы, канцеляризмы;
Убедись, что в готовом тексте не менее трех предложений.
Убедись, что готовый текст начинается с призыва к действию с продуктом.
Убедись, что в готовом тексте есть следующая ключевая информация: Бесплатное обслуживание при покупках от 100 000 рублей в месяц.
"""

# Функция для корректировки использования тире и других символов
def correct_dash_usage(text):
    morph = pymorphy3.MorphAnalyzer()
    text = re.sub(r'\s[-–—]\s', ' — ', text)
    text = re.sub(r'(?<=\d)[-–—](?=\d)', '–', text)
    text = re.sub(r'(?<=[a-zA-Zа-яА-Я0-9])[-–—](?=[a-zA-Zа-яА-Я0-9])', '-', text)
    text = re.sub(r'"([^\"]+)"', r'«\1»', text)
    if text.count('"') == 1:
        text = text.replace('"', '')
    if (text.startswith('"') and text.endswith('"')) or (text.startswith('«') and text.endswith('»')):
        text = text[1:-1].strip()
    text = re.sub(r'(\d+)[kкКK]', r'\1 000', text, flags=re.IGNORECASE)
    greeting_patterns = [
        r"привет\b", r"здравствуй", r"добрый\s(день|вечер|утро)",
        r"дорогой\b", r"уважаемый\b", r"дорогая\b", r"уважаемая\b",
        r"господин\b", r"госпожа\b", r"друг\b", r"коллега\b",
        r"товарищ\b", r"приятель\b", r"подруга\b"
    ]

    def is_greeting_sentence(sentence):
        words = sentence.split()
        if len(words) < 5:
            for word in words:
                parsed = morph.parse(word.lower())[0]
                for pattern in greeting_patterns:
                    if re.search(pattern, parsed.normal_form):
                        return True
        return False

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences and is_greeting_sentence(sentences[0]):
        sentences = sentences[1:]
    text = ' '.join(sentences)

    def restore_yo(text):
        morph = pymorphy3.MorphAnalyzer()
        words = text.split()
        restored_words = []
        for word in words:
            if word.isupper():
                restored_words.append(word)
                continue
            if word.lower() == "все":
                restored_words.append(word)
                continue
            parsed = morph.parse(word)[0]
            restored_word = parsed.word
            if word and word[0].isupper():
                restored_word = restored_word.capitalize()
            restored_words.append(restored_word)
        return ' '.join(restored_words)

    text = restore_yo(text)
    text = re.sub(r'\bИп\b', 'ИП', text, flags=re.IGNORECASE)
    text = re.sub(r'\bОоо\b', 'ООО', text, flags=re.IGNORECASE)
    text = re.sub(r'\bРф\b', 'РФ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bпользовуйтесь\b', 'пользуйтесь', text, flags=re.IGNORECASE)
    text = re.sub(r'\bею\b', 'ей', text, flags=re.IGNORECASE)
    text = re.sub(r'\bповышьте\b', 'повысьте', text, flags=re.IGNORECASE)
    text = re.sub(r'\bСбербизнес\b', 'СберБизнес', text, flags=re.IGNORECASE)
    text = re.sub(r'\bСбербизнеса\b', 'СберБизнес', text, flags=re.IGNORECASE)
    text = re.sub(r'\bСбербизнесе\b', 'СберБизнес', text, flags=re.IGNORECASE)
    text = re.sub(r'\bСбербанк\b', 'СберБанк', text, flags=re.IGNORECASE)
    text = re.sub(r'\bвашего ООО\b', 'вашей компании', text, flags=re.IGNORECASE)
    text = re.sub(r'\b0₽\b', '0 р', text, flags=re.IGNORECASE)
    text = re.sub(r'\b₽\b', 'р', text, flags=re.IGNORECASE)
    text = re.sub(r'\bруб\.(?=\W|$)', 'р', text, flags=re.IGNORECASE)
    text = re.sub(r'\bруб(?:ля|лей)\b', 'р', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s+тысяч(?:а|и)?(?:\s+рублей)?', r'\1 000 р', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*тыс\.\s*руб\.', r'\1 000 р', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*тыс\.\s*р\.', r'\1 000 р', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*тыс\.\s*р', r'\1 000 р', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s+миллиона\b|\bмиллионов\b', r'\1 млн', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*млн\s*руб\.', r'\1 млн р', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*р\b', r'\1 р', text)

    def remove_specific_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        filtered_sentences = [
            sentence for sentence in sentences
            if not re.search(r'\bникаких\s+(посещений|визитов)\b', sentence, re.IGNORECASE)
        ]
        return ' '.join(filtered_sentences)

    text = re.sub(r'\b(\d+)\s+000\s+000\s*р\b', r'\1 млн р', text, flags=re.IGNORECASE)
    text = re.sub(r' р р ', r' р ', text, flags=re.IGNORECASE)
    text = remove_specific_sentences(text)
    return text

# Функция для очистки сообщения
def clean_message(message):
    if not message.endswith(('.', '!', '?')):
        last_period = max(message.rfind('.'), message.rfind('!'), message.rfind('?'))
        if last_period != -1:
            message = message[:last_period + 1]
    return message

# Асинхронная функция для генерации сообщений с использованием abatch
async def generate_messages_in_batches(total_messages, batch_size):
    messages = [prompt_template] * total_messages  # Создаем список одинаковых промптов
    results = []

    # Разбиваем на батчи
    batches = [messages[i:i + batch_size] for i in range(0, total_messages, batch_size)]

    for batch in tqdm_asyncio(batches, desc="Генерация сообщений", unit="batch"):
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                # Отправка батча промптов
                responses = await chat_client.abatch([HumanMessage(content=msg) for msg in batch])
                # Обработка ответов
                for response in responses:
                    if response and response.content:
                        cleaned = clean_message(response.content.strip())
                        corrected = correct_dash_usage(cleaned)
                        results.append(corrected)
                    else:
                        results.append(None)
                break  # Выход из retry цикла при успешном выполнении
            except Exception as e:
                error_message = str(e)
                if "Status 429" in error_message or "Server disconnected without sending a response" in error_message:
                    wait_time = 3
                    print(f"Превышен лимит запросов или сервер не ответил. Ожидание {wait_time} секунд перед повторной попыткой...")
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    print(f"Ошибка при обращении к GigaChat: {e}")
                    # Добавляем None для каждого сообщения в батче в случае непредвиденной ошибки
                    results.extend([None] * len(batch))
                    break
        else:
            print(f"Не удалось обработать батч после {max_retries} попыток.")
            results.extend([None] * len(batch))

    return results

# Основная асинхронная функция
async def main():
    total_messages = 1000
    batch_size = 10  # Размер батча соответствует количеству потоков

    print(f"Начинаем генерацию {total_messages} сообщений с батчами по {batch_size}...")
    start_time = time.time()

    generated_messages = await generate_messages_in_batches(total_messages, batch_size)

    end_time = time.time()
    print(f"Генерация завершена за {end_time - start_time:.2f} секунд.")

    # Фильтрация успешных сообщений
    successful_messages = [msg for msg in generated_messages if msg is not None]
    print(f"Сгенерировано {len(successful_messages)} сообщений из {total_messages}.")

    # Создание DataFrame и сохранение в Excel
    df = pd.DataFrame({'Результат': successful_messages})
    df.to_excel('результаты_генерации.xlsx', index=False)
    print("Результаты сохранены в файл 'результаты_генерации.xlsx'.")

# Запуск основного процесса
if __name__ == "__main__":
    asyncio.run(main())