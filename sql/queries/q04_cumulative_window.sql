-- Задача: как менялась накопленная конверсия по дням. Нужна, чтобы увидеть,
-- стабилизируется ли разница групп или это шум первых дней.
-- Конструкция: оконная функция SUM() OVER (PARTITION BY group ORDER BY date) даёт
-- нарастающий итог и сохраняет все строки. GROUP BY их бы схлопнул.
-- Результат: по строке на день и группу (28 дней x 2 группы = 56 строк).
WITH daily AS (
    SELECT signup_date, group_name,
           COUNT(*)       AS users,
           SUM(converted) AS buyers
    FROM user_analytics
    GROUP BY signup_date, group_name
)
SELECT
    signup_date,
    group_name,
    users,
    buyers,
    SUM(users)  OVER w AS cum_users,
    SUM(buyers) OVER w AS cum_buyers,
    ROUND(SUM(buyers) OVER w * 1.0 / SUM(users) OVER w, 4) AS cum_conversion
FROM daily
WINDOW w AS (PARTITION BY group_name ORDER BY signup_date)
ORDER BY signup_date, group_name;
