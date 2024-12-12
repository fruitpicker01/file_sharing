import asyncio
from langchain.schema import HumanMessage
from langchain_community.chat_models.gigachat import GigaChat
import pandas as pd
import time
import json

# Настройка аутентификации
auth = '...'

# Инициализация клиента GigaChat
chat_client = GigaChat(
    credentials=auth,
    model='GigaChat-Pro',
    scope="GIGACHAT_API_CORP",
    auth_url="https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth",
    max_tokens=68,
    temperature=1.15,
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

# Асинхронная функция для генерации одного сообщения
async def generate_message(semaphore, index, prompt, max_retries=5):
    async with semaphore:
        retries = 0
        while retries < max_retries:
            try:
                # Отправка запроса к GigaChat
                response = await chat_client.ainvoke([HumanMessage(content=prompt)])
                message = response.content.strip()
                # Дополнительная обработка, если требуется
                return message
            except Exception as e:
                error_message = str(e)
                if "Status 429" in error_message or "Server disconnected without sending a response" in error_message:
                    wait_time = 3
                    print(f"[{index}] Превышен лимит запросов или сервер не ответил. Ожидание {wait_time} секунд перед повторной попыткой...")
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    print(f"[{index}] Ошибка при обращении к GigaChat: {e}")
                    return None
        print(f"[{index}] Не удалось получить ответ от GigaChat после {max_retries} попыток.")
        return None

# Основная асинхронная функция для генерации всех сообщений
async def main():
    total_messages = 1000
    concurrency = 10  # Количество одновременных задач
    semaphore = asyncio.Semaphore(concurrency)
    
    tasks = []
    for i in range(total_messages):
        task = asyncio.create_task(generate_message(semaphore, i+1, prompt_template))
        tasks.append(task)
    
    print(f"Запущено {total_messages} задач по генерации сообщений с параллельностью {concurrency}...")
    results = await asyncio.gather(*tasks)
    
    # Фильтрация успешных результатов
    generated_messages = [msg for msg in results if msg is not None]
    print(f"Сгенерировано {len(generated_messages)} сообщений из {total_messages}.")
    
    # Создание DataFrame и сохранение в Excel
    df = pd.DataFrame({'Результат': generated_messages})
    df.to_excel('результаты_генерации.xlsx', index=False)
    print("Результаты сохранены в файл 'результаты_генерации.xlsx'.")

# Запуск основного процесса
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд.")