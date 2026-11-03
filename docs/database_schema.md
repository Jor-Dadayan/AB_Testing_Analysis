# Схема базы данных

experiment_groups (group_id PK, group_name UNIQUE)
  1 -- N users (user_id PK, group_id FK, профиль, корзина)
         1 -- 1 user_outcomes (user_id PK/FK, converted, revenue, purchase_ts)
load_log: журнал загрузок (хеш файла, статус, число строк, ошибка)
user_analytics: одна строка на пользователя, строится из users + outcomes + groups

## Ограничения
- PK и FK, NOT NULL, CHECK на допустимые значения
- ck_buyer_consistency: покупатель имеет revenue > 0 и purchase_ts, остальные имеют 0 и NULL
- Каскадное удаление outcomes при удалении пользователя

## Загрузка (идемпотентная)
CSV -> validate (Python) -> stg_users -> upsert (ON CONFLICT) -> удаление лишних
-> сверка строк. Всё в одной транзакции. Хеш файла в load_log пропускает повторы.

## Файлы sql/
01_schema, 02_upsert_from_staging, 03_user_analytics, 04_integrity_checks, queries/q01-q06
