# ExperimentLab: архитектура

Бизнес-задача: определить, даёт ли новая страница Checkout (Treatment B)
статистически и практически значимый рост конверсии и выручки по сравнению
со старой (Control A).

## Слои
1. Данные: генерация синтетического датасета и проверка качества.
2. Хранилище: PostgreSQL, идемпотентная загрузка, SQL-запросы.
3. Аналитика: SRM, метрики, статтесты, мощность, решение.
4. Представление: графики, ноутбук, Streamlit, итоговый отчёт.

Правило зависимостей: analysis не знает про Streamlit, db не знает про статистику.
Dashboard и ноутбук только вызывают функции из analysis.

## Поток данных
configs/experiment.yaml -> generator -> data/raw/users.csv -> quality checks
-> loader -> PostgreSQL (users, user_outcomes, load_log) -> user_analytics
-> analysis (SRM, метрики, тесты, мощность, решение)
-> notebook / Streamlit / reports

## База данных
- experiment_groups: справочник групп
- users: профиль пользователя, FK на группу
- user_outcomes: покупка и выручка, PK = FK на users
- load_log: журнал загрузок (хеш файла, статус, число строк)
- user_analytics: одна строка на пользователя, источник для анализа
Загрузка идемпотентна: ON CONFLICT и хеш файла. SQL лежит в sql/, а не в Python.

## Модули (src/experimentlab/)
- config.py: чтение YAML и .env
- data/generator.py, data/quality.py
- db/loader.py, db/queries.py
- analysis/srm.py, metrics.py, hypothesis.py, power.py, segments.py, decision.py
- viz/plots.py
- app/dashboard.py

## Стандарты качества
- Типизация и docstrings у публичных функций
- ruff для линтинга, pytest для тестов (статфункции проверяем на данных с известным ответом)
- logging вместо print
- Пароли только в .env
- Коммиты: feat:, fix:, docs:, test:
- Один этап = одна ветка = один pull request

## План этапов
1. Каркас проекта (этот этап)
2. Дизайн эксперимента: размер выборки, MDE, конфиг
3. Генератор данных и проверки качества
4. PostgreSQL: схема, загрузка, SQL
5. Статистический анализ
6. Jupyter Notebook (EDA)
7. Streamlit dashboard
8. Docker Compose для всего проекта, CI
9. README и итоговый вывод для бизнеса

## Критерии готовности проекта
1. docker compose up поднимает систему без ручной настройки
2. Число строк совпадает в CSV, БД и user_analytics
3. Тесты проходят, ruff без ошибок
4. Анализ включает SRM, p-value, доверительные интервалы, размер эффекта, мощность
5. Найденный эффект сопоставлен с заложенным в симуляции
6. В README есть вывод: внедрять дизайн или нет, и почему
7. В репозитории нет паролей и секретных файлов
8. Каждое решение можно объяснить на собеседовании
