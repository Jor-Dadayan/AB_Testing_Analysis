-- Задача: показать правосторонний хвост выручки среди покупателей.
-- Конструкция: WHERE converted = 1 отбирает строки ДО группировки (нули нам не нужны),
-- PERCENTILE_CONT считает медиану и хвостовые перцентили. Если среднее заметно выше
-- медианы, распределение скошено вправо и для выручки нужны устойчивые методы.
-- Результат: 2 строки, у каждой mean > median.
SELECT
    group_name,
    COUNT(*)                                                           AS buyers,
    ROUND(AVG(revenue), 2)                                             AS mean_order,
    ROUND((PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY revenue))::numeric, 2) AS median_order,
    ROUND((PERCENTILE_CONT(0.9)  WITHIN GROUP (ORDER BY revenue))::numeric, 2) AS p90_order,
    ROUND((PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY revenue))::numeric, 2) AS p99_order,
    ROUND(MAX(revenue), 2)                                             AS max_order
FROM user_analytics
WHERE converted = 1
GROUP BY group_name
ORDER BY group_name;
