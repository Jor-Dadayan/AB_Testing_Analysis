# %% [markdown]
# # ExperimentLab: Exploratory Data Analysis
# **End-to-End A/B Testing & Experimentation Analytics Platform**
#
# Разведочный анализ данных A/B-теста новой страницы Checkout.
#
# **Правила этого ноутбука**
# - EDA описывает данные и проверяет их качество. Он **не доказывает** эффект.
# - Ни один график не используется как доказательство значимости. Интервалы и сравнения
#   здесь описательные, проверка гипотез будет на следующем этапе.
# - Скрытый «истинный эффект» симуляции (`data/simulation/`) здесь не читается.
#
# **Содержание:** 1 Business Context, 2 Dataset Overview, 3 Data Types, 4 Missing Values,
# 5 Duplicates, 6 Outliers, 7 Univariate, 8 Bivariate, 9 Multivariate, 10 Correlation,
# 11 Distributions, 12 Control vs Treatment, 13 Data Quality Assessment, 14 Conclusions.

# %% [markdown]
# ## 1. Business Context
#
# Интернет-магазин тестирует новую версию Checkout Page.
# - **Control (A):** старый интерфейс. **Treatment (B):** новый интерфейс.
# - **Единица наблюдения:** пользователь. Группа назначается один раз и не меняется.
# - **Основная метрика:** конверсия (доля пользователей, совершивших покупку).
# - **Вторичные метрики:** выручка на пользователя (RPU) и средний чек покупателей (AOV).
#
# **Вопрос бизнеса:** даёт ли новый дизайн статистически и практически значимый рост
# конверсии и выручки?
#
# **Задачи EDA:** убедиться, что данные чистые и логичные, что рандомизация сработала
# (группы похожи по возрасту, странам, устройствам, каналам), понять форму распределений
# (выручка с множеством нулей и правым хвостом) и сформулировать гипотезы для этапа 5.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from IPython.display import display
from scipy import stats

from experimentlab.analysis import eda
from experimentlab.config import PROJECT_ROOT

warnings.filterwarnings("ignore")
pio.renderers.default = "notebook_connected"
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

PAL = {"control": "#4C72B0", "treatment": "#DD8452"}
GR = eda.GROUPS
ORD = eda.ORDERS
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    """Сохраняет график в reports/figures."""
    fig.savefig(FIG_DIR / f"eda_{name}.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 2. Dataset Overview
# Данные читаются из PostgreSQL (таблица `user_analytics`, одна строка на пользователя).
# Если база недоступна, используется CSV.

# %%
df, SOURCE = eda.load_analytics_frame()
df["log_cart_value"] = np.log(df["cart_value"])
buyers = df[df["converted"] == 1]

print("Источник данных:", SOURCE)
print(f"Строк: {len(df):,} | Столбцов: {df.shape[1]}")
print("Период регистраций:", df["signup_ts"].min(), "->", df["signup_ts"].max())
print(f"Память: {df.memory_usage(deep=True).sum() / 1e6:.1f} МБ")
display(df.head())
display(df.groupby("group_name", observed=True).size().rename("users").to_frame())

# %% [markdown]
# ## 3. Data Types
# Категориальные признаки приведены к `category` с фиксированным порядком, даты к `datetime`,
# деньги к `float` (из PostgreSQL `NUMERIC` приходит как `Decimal`).
#
# | Роль | Столбцы |
# |---|---|
# | Идентификатор | `user_id` |
# | Группа эксперимента | `group_name` |
# | Признаки пользователя (до эксперимента) | `device_type`, `country`, `traffic_source`, `is_new_user`, `age`, `age_group`, `sessions_prior_30d` |
# | Корзина | `cart_items`, `cart_value`, `cart_segment` |
# | Время | `signup_ts`, `signup_date`, `signup_hour`, `day_of_week`, `is_weekend` |
# | Результаты (метрики) | `converted`, `revenue`, `purchase_ts`, `hours_to_purchase` |
#
# Баланс групп проверяется только по признакам «до эксперимента». Результаты сравнивать
# на баланс бессмысленно: они и есть предмет теста.

# %%
types = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_unique": df.nunique(),
    "example": df.iloc[0].astype(str),
})
display(types)

# %% [markdown]
# ## 4. Missing Values Analysis
# Единственный столбец, где пропуски **допустимы по смыслу**: `purchase_ts` и
# `hours_to_purchase` пусты у тех, кто ничего не купил. Остальные пропуски были бы дефектом.

# %%
miss = df.isna().sum().to_frame("missing")
miss["share"] = miss["missing"] / len(df)
display(miss[miss["missing"] > 0])

non_buyers = df["converted"] == 0
print("purchase_ts пуст у всех не купивших:", bool(df.loc[non_buyers, "purchase_ts"].isna().all()))
print("purchase_ts заполнен у всех купивших:", bool(df.loc[~non_buyers, "purchase_ts"].notna().all()))
unexpected = df.drop(columns=["purchase_ts", "hours_to_purchase"]).isna().sum().sum()
print("Неожиданных пропусков:", int(unexpected))

# %% [markdown]
# **Вывод:** пропуски только там, где они логически обязаны быть.
# Заполнять или удалять нечего.
#
# **Ограничение:** синтетические данные не содержат реальных проблем логирования
# (потерянные события, боты). В боевом проекте пропуски были бы отдельной темой.

# %% [markdown]
# ## 5. Duplicate Analysis
# Один пользователь = одна строка. Дубликаты `user_id` сломали бы независимость наблюдений,
# на которой держатся все статистические тесты.

# %%
print("Дубликатов user_id:", int(df["user_id"].duplicated().sum()))
print("Полностью одинаковых строк:", int(df.duplicated().sum()))
print("Одинаковых строк без учёта user_id:", int(df.drop(columns="user_id").duplicated().sum()))

# %% [markdown]
# **Вывод:** дубликатов `user_id` нет, у каждого пользователя ровно одна группа
# (иначе пришлось бы разбираться, какой группе принадлежит наблюдение).
# Редкие совпадения «всех признаков, кроме id» (если есть) случайны: у разных людей могут
# совпасть все значения.

# %% [markdown]
# ## 6. Outlier Detection
# **Исследовательский вопрос:** есть ли экстремальные значения в стоимости корзины, выручке,
# возрасте, и отличаются ли они между группами?
#
# **Почему boxplot:** он показывает медиану, квартили и выбросы по правилу Тьюки
# (за `Q1 - 1.5*IQR` и `Q3 + 1.5*IQR`) и позволяет сравнить группы рядом.
# Для денежных величин используется логарифмическая шкала: иначе хвост «сплющит» коробки.

# %%
rows = {
    "age": eda.iqr_summary(df["age"]),
    "sessions_prior_30d": eda.iqr_summary(df["sessions_prior_30d"]),
    "cart_items": eda.iqr_summary(df["cart_items"]),
    "cart_value": eda.iqr_summary(df["cart_value"]),
    "revenue (покупатели)": eda.iqr_summary(buyers["revenue"]),
}
display(pd.DataFrame(rows).T)

specs = [
    (df, "cart_value", "Стоимость корзины (лог. шкала)", True),
    (buyers, "revenue", "Выручка покупателей (лог. шкала)", True),
    (df, "age", "Возраст", False),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (data, col, title, use_log) in zip(axes, specs, strict=True):
    sns.boxplot(data=data, x="group_name", y=col, hue="group_name", order=GR,
                hue_order=GR, palette=PAL, legend=False, fliersize=2, ax=ax)
    if use_log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("")
fig.tight_layout()
save(fig, "boxplots")
plt.show()

display(buyers.nlargest(5, "revenue")[["user_id", "group_name", "country", "cart_value", "revenue"]])

# %% [markdown]
# **Интерпретация**
# - Возраст и число товаров почти не дают выбросов. Основные «выбросы» в `cart_value`
#   и `revenue` образуются естественным правым хвостом: дорогие корзины встречаются
#   реже, но это реальные покупки, а не ошибки.
# - Коробки двух групп выглядят похожими. Это ожидаемо при рандомизации, но на баланс
#   надо смотреть отдельно (раздел 12).
#
# **Решение:** выбросы **не удаляются**. Это настоящие большие чеки, и удаление исказило бы
# выручку. На следующем этапе нужны методы, устойчивые к хвосту (тест Уэлча, Mann-Whitney,
# bootstrap, при необходимости винзоризация 99-го перцентиля как проверка устойчивости).
#
# **Ограничения:** правило IQR работает плохо для скошенных данных и помечает много
# «нормальных» значений правого хвоста. Boxplot не показывает форму распределения.

# %% [markdown]
# ## 7. Univariate Analysis
# ### 7.1 Histogram (с KDE): возраст и стоимость корзины
# **Исследовательский вопрос:** какова форма распределения возраста и стоимости корзины?
#
# **Почему histogram + KDE:** гистограмма показывает частоты, KDE даёт сглаженную форму и
# помогает увидеть скошенность. Для корзины показаны две шкалы: обычная и логарифмическая.

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
sns.histplot(df["age"], bins=29, kde=True, color="#4C72B0", ax=axes[0])
axes[0].set_title("Возраст")
sns.histplot(df["cart_value"], bins=70, kde=True, color="#55A868", ax=axes[1])
axes[1].set_title("Стоимость корзины (обычная шкала)")
sns.histplot(df["cart_value"], bins=70, kde=True, log_scale=True, color="#55A868", ax=axes[2])
axes[2].set_title("Стоимость корзины (лог. шкала)")
fig.tight_layout()
save(fig, "hist_age_cart")
plt.show()
display(df[["age", "sessions_prior_30d", "cart_items", "cart_value"]].describe().T)

# %% [markdown]
# **Интерпретация:** возраст распределён приблизительно колоколообразно (много людей около
# 30-40 лет). Стоимость корзины имеет выраженный правый хвост: среднее больше медианы;
# на логарифмической шкале форма становится почти симметричной. Это признак
# логнормального распределения, типичного для денежных величин.
#
# **Ограничения:** форма гистограммы зависит от числа корзин (bins), а KDE от ширины
# сглаживания. Возраст ограничен диапазоном 18-75, поэтому обрезан на краях.

# %% [markdown]
# ### 7.2 Countplot: устройства, страны, каналы трафика
# **Исследовательский вопрос:** из кого состоит аудитория? Нет ли сегментов, слишком
# маленьких для анализа?
#
# **Почему countplot:** для категориальных признаков самый прямой способ показать частоты.

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, col in zip(axes, ["device_type", "country", "traffic_source"], strict=True):
    sns.countplot(data=df, x=col, order=ORD[col], color="#6A8CAF", ax=ax)
    for p in ax.patches:
        ax.annotate(f"{p.get_height() / len(df):.0%}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_title(col)
    ax.set_xlabel("")
fig.tight_layout()
save(fig, "countplots")
plt.show()

# %% [markdown]
# **Интерпретация:** преобладает mobile, самая крупная страна US, каналы распределены
# неравномерно, tablet и email относительно малы. Малые сегменты (tablet, email,
# отдельные страны) при анализе по сегментам дадут широкие доверительные интервалы.
#
# **Ограничения:** доли описывают этот эксперимент, а не всех посетителей магазина.
# Категория `OTHER` объединяет разные страны.

# %% [markdown]
# ## 8. Bivariate Analysis
# ### 8.1 Конверсия по сегментам
# **Исследовательский вопрос:** как конверсия связана с устройством, каналом и новизной
# пользователя (без деления на группы)?
#
# **Почему столбчатая диаграмма:** конверсия это доля, а столбцы удобно сравнивать со
# средней линией по всей выборке.

# %%
overall = df["converted"].mean()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, col in zip(axes, ["device_type", "traffic_source", "is_new_user"], strict=True):
    s = df.groupby(col, observed=True)["converted"].mean()
    s.plot.bar(ax=ax, color="#6A8CAF", rot=20)
    ax.axhline(overall, ls="--", c="gray", lw=1)
    for i, v in enumerate(s):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Конверсия по: {col}")
    ax.set_xlabel("")
fig.tight_layout()
save(fig, "conversion_by_segment")
plt.show()

# %% [markdown]
# **Интерпретация:** конверсия заметно зависит от канала, устройства и новизны пользователя
# (email и desktop выше, social и новые пользователи ниже средней линии). Значит,
# состав аудитории в группах важен: перекос по этим признакам исказил бы сравнение.
#
# **Ограничения:** это связи, а не причины. Каналы коррелируют с устройством и новизной,
# поэтому отдельный столбец смешивает несколько влияний (см. раздел 9).

# %% [markdown]
# ### 8.2 Violinplot: стоимость корзины по устройствам и группам
# **Исследовательский вопрос:** отличается ли распределение стоимости корзины между
# устройствами и совпадает ли оно в Control и Treatment?
#
# **Почему violin (split):** показывает всю форму распределения, а не только квартили.
# Половинки скрипки по группам дают прямое сравнение. Использована лог. шкала и
# случайная выборка 20 000 строк для скорости.

# %%
sample = df.sample(20_000, random_state=0)
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=sample, x="device_type", y="log_cart_value", hue="group_name",
               split=True, inner="quart", cut=0, order=ORD["device_type"],
               hue_order=GR, palette=PAL, ax=ax)
ax.set_ylabel("ln(стоимость корзины)")
ax.set_xlabel("")
ax.set_title("Стоимость корзины по устройствам и группам")
fig.tight_layout()
save(fig, "violin_cart_device")
plt.show()

# %% [markdown]
# **Интерпретация:** корзины desktop сдвинуты вверх относительно mobile. Левая и правая
# половины каждой скрипки очень похожи: распределения корзины в группах совпадают.
#
# **Ограничения:** показана выборка, а логарифм меняет масштаб оси (разницы на графике
# это отношения, а не доллары).

# %% [markdown]
# ### 8.3 Scatterplot: корзина и выручка покупателей
# **Исследовательский вопрос:** насколько выручка зависит от стоимости корзины?
#
# **Почему scatter:** это связь двух числовых величин. Нужны лог. шкалы по обеим осям
# и выборка 5 000 покупателей, иначе точки сольются.

# %%
sc = buyers.sample(min(5_000, len(buyers)), random_state=0)
fig, ax = plt.subplots(figsize=(6.5, 6))
sns.scatterplot(data=sc, x="cart_value", y="revenue", hue="group_name", hue_order=GR,
                palette=PAL, alpha=0.4, s=14, ax=ax)
lim = [sc["cart_value"].min(), sc["cart_value"].max()]
ax.plot(lim, lim, "k--", lw=1, label="y = x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_title("Выручка и стоимость корзины (покупатели)")
fig.tight_layout()
save(fig, "scatter_cart_revenue")
plt.show()
print("Spearman:", round(buyers[["cart_value", "revenue"]].corr(method="spearman").iloc[0, 1], 3))

# %% [markdown]
# **Интерпретация:** точки лежат вокруг диагонали: выручка примерно равна стоимости корзины
# с разбросом. Группы перемешаны и визуально неразличимы.
#
# **Ограничения:** график охватывает только покупателей (у остальных выручка 0), поэтому
# он ничего не говорит о конверсии. Визуально различить небольшой сдвиг среднего чека
# между группами здесь невозможно, для этого нужен тест.

# %% [markdown]
# ## 9. Multivariate Analysis
# ### 9.1 Heatmap: конверсия по устройству и каналу
# **Исследовательский вопрос:** как конверсия зависит от сочетания устройства и канала, и
# хватает ли пользователей в каждой ячейке?
#
# **Почему heatmap с двумя панелями:** слева конверсия, справа размер ячейки. Конверсия
# в маленькой ячейке ненадёжна, и это видно сразу.

# %%
rate = pd.crosstab(df["device_type"], df["traffic_source"], values=df["converted"], aggfunc="mean")
cnt = pd.crosstab(df["device_type"], df["traffic_source"])
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.heatmap(rate, annot=True, fmt=".1%", cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Конверсия")
sns.heatmap(cnt, annot=True, fmt=",d", cmap="Greys", ax=axes[1])
axes[1].set_title("Число пользователей")
fig.tight_layout()
save(fig, "heatmap_device_source")
plt.show()

# %% [markdown]
# **Интерпретация:** конверсия меняется по обеим осям, а размеры ячеек очень разные
# (tablet с email и social мал). Поэтому сегментный анализ надо вести осторожно.
#
# **Ограничения:** 15 ячеек = 15 потенциальных сравнений. Если искать «красивую» ячейку,
# её можно найти случайно (проблема множественных сравнений).

# %% [markdown]
# ### 9.2 Plotly: конверсия по устройству, новизне и группе
# **Исследовательский вопрос:** одинаково ли ведут себя группы у новых и вернувшихся
# пользователей на разных устройствах?
#
# **Почему интерактивный Plotly:** три измерения (устройство, новизна, группа) удобно
# читать с подсказками при наведении, включая размер каждой подгруппы.

# %%
agg = (df.groupby(["device_type", "is_new_user", "group_name"], observed=True)["converted"]
       .agg(conversion="mean", users="size").reset_index())
agg["is_new_user"] = agg["is_new_user"].map({0: "вернувшийся", 1: "новый"})
fig = px.bar(agg, x="device_type", y="conversion", color="group_name", barmode="group",
             facet_col="is_new_user", color_discrete_map=PAL, hover_data=["users"],
             category_orders={"device_type": ORD["device_type"], "group_name": GR},
             title="Конверсия: устройство x новизна x группа")
fig.update_yaxes(tickformat=".0%")
fig.show()

# %% [markdown]
# **Интерпретация:** вернувшиеся пользователи конвертируются чаще новых на всех устройствах.
# Внутри каждой пары столбцов (Control и Treatment) разница небольшая по сравнению
# с различиями между сегментами.
#
# **Ограничения:** столбцы без интервалов. Различия внутри подгрупп могут быть шумом,
# особенно у tablet. Это гипотеза для этапа 5, а не вывод.

# %% [markdown]
# ## 10. Correlation Analysis
# **Исследовательский вопрос:** какие признаки связаны друг с другом и с результатом?
#
# **Почему корреляция Спирмена + heatmap:** признаки скошены (корзина, выручка) и частично
# бинарные. Спирмен основан на рангах, не требует нормальности и не боится выбросов.

# %%
corr_cols = ["age", "sessions_prior_30d", "cart_items", "cart_value", "is_new_user",
             "is_weekend", "signup_hour", "converted", "revenue"]
corr = df[corr_cols].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Корреляция Спирмена")
fig.tight_layout()
save(fig, "correlation_heatmap")
plt.show()

pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("rho").to_frame()
pairs["abs_rho"] = pairs["rho"].abs()
display(pairs.sort_values("abs_rho", ascending=False).head(8).drop(columns="abs_rho"))

# %% [markdown]
# **Интерпретация:** сильнее всего связаны `converted` и `revenue` (выручка бывает только у
# покупателей), а также `is_new_user` и `sessions_prior_30d` (у новых почти нет прошлых
# сессий), `cart_items` и `cart_value`. Остальные связи слабые.
#
# **Ограничения:** корреляция не доказывает причинность. Связь `converted` и `revenue`
# тривиальна по построению. Спирмен улавливает только монотонные связи. Нули в выручке
# искажают её корреляции. Корреляций с группой здесь нет: баланс проверяется в разделе 12.

# %% [markdown]
# ## 11. Distribution Analysis
# ### 11.1 Revenue Distribution
# **Исследовательский вопрос:** какова форма выручки, какая доля нулей и подходят ли
# методы, рассчитанные на нормальное распределение?
#
# **Почему три панели:** (1) доля нулей, (2) распределение положительной выручки на
# лог. шкале, (3) Q-Q график логарифма: прямая линия означает логнормальность.

# %%
zero_share = (df["revenue"] == 0).mean()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].bar(["revenue = 0", "revenue > 0"], [zero_share, 1 - zero_share],
            color=["#B0B0B0", "#4C72B0"])
for i, v in enumerate([zero_share, 1 - zero_share]):
    axes[0].text(i, v, f"{v:.1%}", ha="center", va="bottom")
axes[0].set_title("Доля нулевой выручки")
sns.histplot(buyers["revenue"], bins=70, log_scale=True, kde=True, color="#4C72B0", ax=axes[1])
axes[1].set_title("Выручка покупателей (лог. шкала)")
stats.probplot(np.log(buyers["revenue"]), dist="norm", plot=axes[2])
axes[2].set_title("Q-Q график ln(выручки покупателей)")
fig.tight_layout()
save(fig, "revenue_distribution")
plt.show()


def describe(s: pd.Series) -> pd.Series:
    return pd.Series({"mean": s.mean(), "median": s.median(), "std": s.std(),
                      "skew": s.skew(), "kurtosis": s.kurt(),
                      "p90": s.quantile(0.9), "p99": s.quantile(0.99), "max": s.max()})


display(pd.DataFrame({"все пользователи": describe(df["revenue"]),
                      "только покупатели": describe(buyers["revenue"])}))

# %% [markdown]
# **Интерпретация:** около 9 из 10 пользователей имеют нулевую выручку. У покупателей
# распределение скошено вправо (среднее больше медианы, большая асимметрия), а на лог. шкале
# оно близко к колоколу и Q-Q график близок к прямой, то есть выручка примерно логнормальна.
#
# **Следствие для анализа:** выручка на пользователя это смесь «много нулей + тяжёлый хвост».
# t-тест на сырых значениях здесь хрупок. Нужны: тест Уэлча (при больших выборках среднее
# всё равно приближается к нормальному), Mann-Whitney, bootstrap, а также раздельный
# анализ «конверсия» и «средний чек среди покупателей».
#
# **Ограничения:** Q-Q график показывает отклонения в хвостах плохо при больших n.
# Любые тесты на нормальность при 100 000 строк отвергают всё, поэтому здесь они не нужны.

# %% [markdown]
# ### 11.2 KDE: распределения по группам
# **Исследовательский вопрос:** совпадают ли формы распределений возраста и выручки
# покупателей в Control и Treatment?
#
# **Почему KDE с `common_norm=False`:** плотности сравнимы, даже если группы разного
# размера. Для выручки лог. шкала.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
sns.kdeplot(data=df, x="age", hue="group_name", hue_order=GR, palette=PAL, fill=True,
            alpha=0.3, common_norm=False, ax=axes[0])
axes[0].set_title("Возраст по группам")
sns.kdeplot(data=buyers, x="revenue", hue="group_name", hue_order=GR, palette=PAL, fill=True,
            alpha=0.3, common_norm=False, log_scale=True, ax=axes[1])
axes[1].set_title("Выручка покупателей по группам (лог. шкала)")
fig.tight_layout()
save(fig, "kde_by_group")
plt.show()

# %% [markdown]
# **Интерпретация:** кривые возраста практически накладываются друг на друга. Кривые
# выручки очень близки. Даже если есть небольшой сдвиг, на глаз его не отличить от шума.
#
# **Ограничения:** KDE сглаживает и скрывает мелкие различия. Сравнение средних
# невозможно сделать визуально: нужен формальный тест (этап 5).

# %% [markdown]
# ## 12. Control vs Treatment Comparison
# ### 12.1 Баланс групп (особое внимание: возраст, страна, устройство, канал)
# **Исследовательский вопрос:** одинаков ли состав групп до воздействия?
# Если рандомизация сработала, любые различия в исходе можно приписать дизайну, а не составу.
#
# **Метод:** стандартизованная разница (SMD) для каждого признака. Она не зависит от
# размера выборки. Ориентир: |SMD| < 0.1 означает баланс.
# p-значения показаны справочно: при множестве проверок около 5% дадут p < 0.05 просто
# случайно, и опираться надо на SMD.

# %%
balance = eda.balance_table(
    df,
    numeric=["age", "sessions_prior_30d", "cart_items", "cart_value", "signup_hour"],
    categorical=["country", "device_type", "traffic_source", "is_new_user",
                 "age_group", "is_weekend", "cart_segment"],
)
display(balance)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#C44E52" if not b else "#55A868" for b in balance["balanced"]]
ax.barh(balance["variable"], balance["abs_smd"], color=colors)
ax.axvline(0.1, ls="--", c="black", lw=1)
ax.invert_yaxis()
ax.set_xlabel("|SMD| (порог 0.1)")
ax.set_title("Баланс Control и Treatment по признакам до эксперимента")
fig.tight_layout()
save(fig, "balance_smd")
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ["age_group", "country", "device_type", "traffic_source"],
                   strict=True):
    shares = pd.crosstab(df[col], df["group_name"], normalize="columns").reindex(ORD[col])
    shares.plot.bar(ax=ax, color=[PAL[c] for c in shares.columns], rot=0)
    ax.set_title(f"Доли по {col} внутри групп")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
fig.tight_layout()
save(fig, "balance_shares")
plt.show()

# %% [markdown]
# **Интерпретация:** при корректной рандомизации ожидаются |SMD| ниже 0.1 по всем признакам
# и почти одинаковые столбцы Control и Treatment в каждой категории. Проверьте таблицу:
# если какой-то признак окрашен красным, это сигнал расследовать (разобраться, нет ли
# ошибки в назначении групп), а не повод менять данные.
#
# **Ограничения:** баланс по наблюдаемым признакам не гарантирует баланс по ненаблюдаемым.
# Проверка пересчёта долей групп (SRM) формально делается на этапе 5.

# %% [markdown]
# ### 12.2 Conversion Rate Comparison
# **Исследовательский вопрос:** как различается наблюдаемая конверсия двух групп?
#
# **Почему столбцы с 95% интервалами Уилсона:** интервалы показывают неопределённость
# оценки доли. Если интервалы почти не пересекаются, это намёк, но **не** тест гипотезы.

# %%
cr = eda.conversion_ci(df, "group_name")
display(cr)
c, t = cr.loc["control", "rate"], cr.loc["treatment", "rate"]
print(f"Абсолютная разница: {t - c:+.4f} ({(t - c) * 100:+.2f} п.п.)")
print(f"Относительная разница: {(t / c - 1) * 100:+.1f}%")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.bar(cr.index.astype(str), cr["rate"], color=[PAL[g] for g in cr.index],
       yerr=[cr["rate"] - cr["ci_low"], cr["ci_high"] - cr["rate"]], capsize=8)
for i, v in enumerate(cr["rate"]):
    ax.text(i, v / 2, f"{v:.2%}", ha="center", color="white", fontweight="bold")
ax.set_ylabel("Конверсия")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Конверсия по группам (95% интервал Уилсона)")
fig.tight_layout()
save(fig, "conversion_by_group")
plt.show()

# %%
cd = eda.conversion_ci(df, ["device_type", "group_name"]).reset_index()
cd["err_up"] = cd["ci_high"] - cd["rate"]
cd["err_down"] = cd["rate"] - cd["ci_low"]
fig = px.bar(cd, x="device_type", y="rate", color="group_name", barmode="group",
             error_y="err_up", error_y_minus="err_down", color_discrete_map=PAL,
             category_orders={"device_type": ORD["device_type"], "group_name": GR},
             hover_data=["count"], title="Конверсия по устройствам и группам (95% интервал)")
fig.update_yaxes(tickformat=".0%")
fig.show()

# %% [markdown]
# **Интерпретация:** в таблице выше видна наблюдаемая разница конверсий. Это разница
# **в выборке**: она включает случайный шум. Интервалы показывают, насколько оценки неточны,
# особенно в малых сегментах (tablet).
#
# **Чего нельзя утверждать:** что разница значима, что она одинакова во всех сегментах,
# что она практически важна. Для этого нужны z-тест, интервал для разницы и порог
# практической значимости (этап 5).
#
# **Ограничения:** интервалы Уилсона построены для каждой группы отдельно, а для разницы
# нужен свой интервал. Сегментные столбцы подвержены проблеме множественных сравнений.

# %% [markdown]
# ### 12.3 Выручка: RPU, средний чек и доля нулей по группам
# **Исследовательский вопрос:** как различаются метрики выручки?
#
# **Почему три панели:** RPU = конверсия x средний чек. Разложение показывает, откуда
# берётся разница: из покупательской активности или из размера чека.
# Усы это ±1.96 стандартной ошибки среднего (описательно).

# %%
rev = df.groupby("group_name", observed=True).agg(
    users=("user_id", "size"),
    rpu=("revenue", "mean"),
    rpu_se=("revenue", "sem"),
    zero_share=("revenue", lambda s: (s == 0).mean()),
)
aov = buyers.groupby("group_name", observed=True)["revenue"].agg(
    aov="mean", aov_se="sem", median_order="median")
rev = rev.join(aov)
display(rev)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
cols = [PAL[g] for g in rev.index]
axes[0].bar(rev.index.astype(str), rev["rpu"], yerr=1.96 * rev["rpu_se"], capsize=6, color=cols)
axes[0].set_title("Выручка на пользователя (RPU)")
axes[1].bar(rev.index.astype(str), rev["aov"], yerr=1.96 * rev["aov_se"], capsize=6, color=cols)
axes[1].set_title("Средний чек покупателей (AOV)")
axes[2].bar(rev.index.astype(str), rev["zero_share"], color=cols)
axes[2].set_ylim(0.8, 0.95)
axes[2].set_title("Доля пользователей без покупки (ось обрезана)")
fig.tight_layout()
save(fig, "revenue_by_group")
plt.show()

# %% [markdown]
# **Интерпретация:** RPU определяется в основном долей покупателей, средний чек меняется
# мало. Усы RPU и AOV широкие из-за тяжёлого хвоста.
#
# **Ограничения:** у третьей панели обрезана ось (иначе разницу не видно), поэтому на глаз
# она выглядит больше, чем есть. Стандартная ошибка среднего для скошенной выручки
# приближённая, нужен bootstrap.

# %% [markdown]
# ### 12.4 Time Series Analysis
# **Исследовательский вопрос:** одинаково ли группы набирались по дням, и как менялась
# накопленная конверсия (стабилизируется ли разница или это шум первых дней)?
#
# **Почему линии по дням:** ровные параллельные кривые числа пользователей
# говорят об исправности рандомизации. Накопленная конверсия показывает, как оценка
# «сходится». Резкий спад со временем мог бы означать эффект новизны.

# %%
daily = (df.groupby(["signup_date", "group_name"], observed=True)
         .agg(users=("user_id", "size"), buyers=("converted", "sum"))
         .reset_index().sort_values(["group_name", "signup_date"]))
daily["conversion"] = daily["buyers"] / daily["users"]
by_g = daily.groupby("group_name", observed=True)
daily["cum_conversion"] = by_g["buyers"].cumsum() / by_g["users"].cumsum()

fig = px.line(daily, x="signup_date", y="users", color="group_name", markers=True,
              color_discrete_map=PAL, title="Новые пользователи по дням")
fig.show()
fig = px.line(daily, x="signup_date", y="cum_conversion", color="group_name",
              color_discrete_map=PAL, title="Накопленная конверсия по дням")
fig.update_yaxes(tickformat=".1%")
fig.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
hourly = pd.crosstab(df["signup_hour"], df["group_name"])
hourly.plot(ax=axes[0], color=[PAL[c] for c in hourly.columns], marker="o", ms=3)
axes[0].set_title("Регистрации по часу суток")
sns.countplot(data=df, x="day_of_week", hue="group_name", hue_order=GR, palette=PAL, ax=axes[1])
axes[1].set_xticks(range(7), ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
axes[1].set_xlabel("")
axes[1].set_title("Регистрации по дням недели")
fig.tight_layout()
save(fig, "time_patterns")
plt.show()

# %% [markdown]
# **Интерпретация:** число пользователей по дням в группах идёт параллельно, пики по
# часам (обед и вечер) и более высокий трафик в выходные одинаковы для обеих групп.
# Накопленная конверсия сильно колеблется в первые дни и затем стабилизируется.
# Это значит, что смотреть на результат в первые дни (peeking) рискованно.
#
# **Ограничения:** период всего 28 дней, поэтому долгосрочный эффект новизны оценить
# нельзя. Дневная конверсия шумная из-за малых дневных выборок.

# %% [markdown]
# ## 13. Data Quality Assessment
# Сводная карточка: каждая проверка даёт PASS или WARN. WARN это повод разобраться,
# а не автоматически «данные плохие».

# %%
share_t = (df["group_name"] == "treatment").mean()
checks = {
    "user_id уникален": df["user_id"].is_unique,
    "нет полностью одинаковых строк": not df.duplicated().any(),
    "нет неожиданных пропусков": unexpected == 0,
    "выручка > 0 только у покупателей": bool(((df["revenue"] > 0) == (df["converted"] == 1)).all()),
    "выручка неотрицательна": bool((df["revenue"] >= 0).all()),
    "покупка не раньше регистрации": bool((df["hours_to_purchase"].dropna() >= 0).all()),
    "возраст в диапазоне 18-75": bool(df["age"].between(18, 75).all()),
    "доли групп близки к 50/50 (отклонение < 0.5 п.п.)": abs(share_t - 0.5) < 0.005,
    "баланс ковариат: все |SMD| < 0.1": bool(balance["balanced"].all()),
    "нет пустых сегментов устройство x канал": bool((cnt > 0).all().all()),
}
quality = pd.DataFrame({"проверка": list(checks), "результат":
                        ["PASS" if v else "WARN" for v in checks.values()]})
display(quality)
print(f"Итого: {(quality['результат'] == 'PASS').sum()}/{len(quality)} проверок PASS")

# %% [markdown]
# **Оценка качества:** данные пригодны для анализа, если все проверки PASS.
#
# **Ограничения этих данных (важно сказать на собеседовании):**
# - Данные синтетические: нет ботов, дублей событий, потерянных логов, пропусков.
# - Эффект в симуляции однороден по сегментам и не зависит от времени (нет эффекта новизны).
# - Нет информации о ненаблюдаемых факторах, поэтому баланс проверен только по известным
#   признакам.
# - 28 дней не покрывают сезонность и долгосрочное поведение.

# %% [markdown]
# ## 14. EDA Conclusions
# ### Основные наблюдения
# 1. **Структура:** одна строка на пользователя, дубликатов нет, пропуски только там, где
#    они логически обязаны быть (`purchase_ts` у не купивших).
# 2. **Рандомизация:** группы близки по размеру, возрасту, странам, устройствам, каналам и
#    новизне (см. таблицу SMD в 12.1). Состав аудитории по дням и часам одинаков.
# 3. **Конверсия** сильно зависит от устройства, канала и новизны пользователя
#    (мощные ковариаты). Размеры сегментов очень различаются.
# 4. **Выручка:** доля нулей около 90%, у покупателей скос вправо, на логарифмической шкале
#    распределение близко к нормальному (логнормальность).
# 5. **Выбросы** в стоимости корзины и выручке это настоящий правый хвост, а не ошибки.
#    Их нельзя удалять.
# 6. **Наблюдаемая разница** групп в конверсии и выручке есть в таблице 12.2 и 12.3, но это
#    разница выборки: значимость пока не проверена.
# 7. **Время:** накопленная конверсия шумит в первые дни, поэтому нельзя принимать решение
#    по ранним данным.
#
# ### Гипотезы и задачи для этапа 5 (статистический анализ)
# | # | Что проверить | Метод |
# |---|---|---|
# | H0 | Нет ли перекоса размера групп (SRM) | Хи-квадрат на долях групп |
# | H1 | Конверсия Treatment выше Control | Z-тест двух долей, интервал для разницы, relative lift |
# | H2 | RPU Treatment выше Control | Тест Уэлча, Mann-Whitney, bootstrap-интервал |
# | H3 | Средний чек среди покупателей изменился | Тест на лог-выручке, bootstrap |
# | H4 | Эффект практически значим | Порог (MDE) из дизайна эксперимента, сравнение с интервалом |
# | H5 | Эффект одинаков в сегментах (устройство, канал, новизна) | Анализ по сегментам с поправкой на множественные сравнения (Холм/Бенджамини-Хохберг) |
# | H6 | Эффект остаётся после учёта ковариат | Логистическая регрессия: converted ~ group + device + source + is_new_user + ... |
# | H7 | Нет эффекта новизны / нестабильности | Конверсия по неделям, динамика разницы |
# | H8 | Хватает ли мощности | Расчёт мощности и MDE для 100 000 пользователей |
# | H9 | Результат устойчив к хвосту | Винзоризация 99-го перцентиля, лог-преобразование |
#
# ### Что нельзя утверждать после EDA
# EDA не отвечает на вопрос «внедрять или нет». Он показывает, что данные пригодны для
# сравнения и **какие методы** для этого нужны.
