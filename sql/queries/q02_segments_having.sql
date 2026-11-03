-- Задача: конверсия по сегментам устройство x канал x группа, без мелких сегментов.
-- Конструкция: HAVING фильтрует уже сгруппированные строки (WHERE работает до группировки).
-- Так отсекаются сегменты, где мало пользователей и цифры случайны.
-- Результат: несколько десятков строк, сегменты меньше 1000 пользователей отсутствуют.
SELECT
    device_type,
    traffic_source,
    group_name,
    COUNT(*)                          AS users,
    ROUND(AVG(converted::numeric), 4) AS conversion,
    ROUND(AVG(revenue), 2)            AS revenue_per_user
FROM user_analytics
GROUP BY device_type, traffic_source, group_name
HAVING COUNT(*) >= 1000
ORDER BY device_type, traffic_source, group_name;
