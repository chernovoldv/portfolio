import numpy as np
import pandas as pd
import pandahouse as ph
import requests

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import io
import datetime
import os

from airflow.decorators import dag, task

default_args = {
    'owner': 'd-chernovol',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': datetime.datetime(2023, 12, 23)
}


schedule_interval = '*/15 * * * *'

# создаем подключение к БД, host и пароль хранятся в Variables (Airflow -> CI/CD)
connection = {'host': os.environ.get("host_connection"),
'database':'simulator',
'user':'student',
'password': os.environ.get('db_password')
}

def extract_from_db(query: str, types: dict):
    '''
    Функция для выполнения запроса.\n
    Параметры:\n
    query: str\n
        SQL - запрос\n
    types: dict\n
        Словарь с типами данных
    '''
    df = ph.read_clickhouse(query=query, connection=connection).astype(types)
    return df


def find_anomalies(df: pd.core.frame.DataFrame, metric: str, days: int, periods: int, rolling_n: int, days_alpha: float, periods_alpha: float, rolling_func: str = 'mean' or 'median', time_column: str ='time_period'):
    '''
    Функция для поиска аномальных значениий.\n
    Будет использована внутри make_anomalies_report\n
    Значение будет считать аномальным, если оно значительно откличается от значений метрик за последние несколько периодов и значений за прошлые дни.\n
    Параметры:\n
    df: pd.core.frame.DataFrame
        DataFrame с метриками\n
    metric: str\n
        Название метрики из датафрейма\n
    days: int\n
        Количество дней для расчета ДИ на основе межквартильного размаха\n
        Например, берется значение метрики за 15 минутный интервал сегодня и сутки назад
    periods: int\n
        Количество периодов для расчета ДИ на основе межквартильного размаха\n
        Например, последний час = 15 минут * 4, 4 - это period\n
    rolling_n: int\n
        Границы ДИ сглаживаются за указанный период (rolling за 5 периодов с центрированием)\n
    days_alpha: float\n
        Задаем коэффициент Alpha для расчета ДИ для дней\n
    periods_alpha: float\n
        Задаем коэффициент Alpha для расчета ДИ для периодов\n
    rolling_func: str = 'mean' or 'median'\n
        Для сглаживания границ ДИ можно использовать средние значения или медиану.\n
        При выборе медианы аномальные значения прошлых периодов меньше влияют на ширину ДИ\n
    '''
    # копируем датафрейм
    df_copy = df.copy().sort_values(by=time_column)
    # добавляем столбцы для доверительного интервала и пометки выбросов
    df_copy[f'anomaly_{metric}'] = ''
    df_copy[f'anomaly_{metric}_CI_up'] = ''
    df_copy[f'anomaly_{metric}_CI_down'] = ''
    # смещение значений метрик
    df_copy[f'{metric}_shift'] = df_copy[f'{metric}'].shift(1)
    # для сглаживание границ ДИ может быть  использовано среднее значение или медиана
    if rolling_func != 'mean' and rolling_func != 'median':
        return print('Rolling func error: imput "mean" or "median"')
    # итерация
    for i, row in df_copy.iterrows():
        # берем даты для построения ДИ на основе данные за прошлые дни
        days_for_check = [df_copy['time_period'][i] - datetime.timedelta(days=int(d)) for d in np.arange(1, days +1)]
        # считаем квантили
        q_25_d = df_copy[df_copy['time_period'].isin(days_for_check)][f'{metric}_shift'].quantile(0.25)
        q_75_d = df_copy[df_copy['time_period'].isin(days_for_check)][f'{metric}_shift'].quantile(0.75)
        # считаем границы ДИ по дням
        ci_days_upper = q_75_d + days_alpha * (q_75_d - q_25_d)
        ci_days_down = q_25_d - days_alpha * (q_75_d - q_25_d)
        # берем прошлые периоды (по 15 минут) для расчета ДИ на основе предыдущих интервалов
        periods_for_check = [df_copy['time_period'][i] - datetime.timedelta(minutes=int(p)) for p in np.arange(15, periods * 15 + 1, 15)]
        # считаем квантили
        q_25_m = df_copy[df_copy['time_period'].isin(periods_for_check)][f'{metric}_shift'].quantile(0.25)
        q_75_m = df_copy[df_copy['time_period'].isin(periods_for_check)][f'{metric}_shift'].quantile(0.75)
        # считаем границы ДИ по прошлым интервалам
        ci_periods_upper = q_75_m + periods_alpha * (q_75_m - q_25_m)
        ci_periods_down = q_25_m - periods_alpha * (q_75_m - q_25_m)
        # значение будет считаться аномальным, если оно не попадает в оба ДИ, поэтому берем максимальный верхний и минимальный нижний интервалы
        df_copy.at[i, f'anomaly_{metric}_CI_up'] = max(ci_days_upper, ci_periods_upper)
        df_copy.at[i, f'anomaly_{metric}_CI_down'] = min(ci_days_down, ci_periods_down)
    # расчет ДИ со сглаживанием по медианам или средним значениям
    if rolling_func == 'median':
        df_copy[f'anomaly_{metric}_CI_up'] = df_copy[f'anomaly_{metric}_CI_up'].rolling(rolling_n, center=True, min_periods=1).median()
        df_copy[f'anomaly_{metric}_CI_down'] = df_copy[f'anomaly_{metric}_CI_down'].rolling(rolling_n, center=True, min_periods=1).median()
    elif rolling_func == 'mean':
        df_copy[f'anomaly_{metric}_CI_up'] = df_copy[f'anomaly_{metric}_CI_up'].rolling(rolling_n, center=True, min_periods=1).mean()
        df_copy[f'anomaly_{metric}_CI_down'] = df_copy[f'anomaly_{metric}_CI_down'].rolling(rolling_n, center=True, min_periods=1).mean()
    # Проставляем метки для выбросов
    for i, row in df_copy.iterrows():
        if (row[metric] < row[f'anomaly_{metric}_CI_down']) or (row[metric] > row[f'anomaly_{metric}_CI_up']):
            df_copy.at[i, f'anomaly_{metric}'] = True
        else:
            df_copy.at[i, f'anomaly_{metric}'] = False

    return df_copy


def make_message(metric_name: str, current_value: int or float, prev_value: int or float, query_name: str, time, dashboard: str, time_period: str = '15 минут'):
    '''
    Функция для создания сообщения.\n
    Расчет отклонения в %, функция будет использована внутри make_anomalies_report.\n
    Параметры:\n
    metric_name: str\n
        Название метрики\n
    current_value: int or float\n
        Текущее значение метрики\n
    prev_value: int or float\n
        Предыдущее значение метрики\n
    query_name: str\n
        Название запроса\n
    time\n
        Время для текущего значения метрики\n
    time_period: str = '15 минут'\n
        Временной отрезок, по-умолчанию 15 минут
    '''
    x = round((current_value/prev_value - 1)*100, 2)
    message = f'''Метрика {metric_name} - {query_name} в срезе {time_period}, время {time}.
Текущее значение {current_value:.2f}. Отклонение более {x:.2f}%.
Dashboard: {dashboard}
'''
    return message


def make_graph(df: pd.core.frame.DataFrame, metric: str, metric_name:str, query_name: str):
    '''
    Функция для создания графиков.\n
    Функция будет использована внутри make_anomalies_report.\n
    Параметры:\n
    df: pd.core.frame.DataFrame\n
        Dataframe со значениями метрики\n
    metric: str\n
        Название колонки с метриков в df\n
    metric_name:str\n
        Название метрики\n
    query_name: str\n
        Название запроса\n
    '''
    # определяем размер графика
    fig, ax = plt.subplots(figsize=(10, 7))
    # строим верхнюю границу ДИ (прозрачную)
    sns.lineplot(
        data=df,
        y = f'anomaly_{metric}_CI_up',
        x = 'time_period',
        ax=ax,
        alpha=0
    )
    # строим нижнюю границу ДИ (прозрачную)
    sns.lineplot(
        data=df,
        y = f'anomaly_{metric}_CI_down',
        x = 'time_period',
        ax=ax,
        alpha=0
    )
    # ДИ будет закрашен
    ax.fill_between(
        ax.get_lines()[0].get_xdata(),
        ax.get_lines()[0].get_ydata(),
        ax.get_lines()[1].get_ydata(),
        color='purple', alpha=0.3, label='ДИ')
    # Строим диаграмму рассеяния со значениями метрик
    sns.scatterplot(
        data = df,
        y = metric,
        x = 'time_period',
        hue=f'anomaly_{metric}',
        hue_order=[True, False],
        palette = ['red', 'blue'],
        ax=ax
    )
    # настраиваем формат отображения даты и времени на оси X
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.title(f'{metric_name} - {query_name}', fontsize=20)

    if abs(df[metric].max()) <= 1 and abs(df[metric].min()) <= 1:
        plt.ylabel('Показатель')
    else:
        plt.ylabel('Количество')
    plt.xlabel('Время')
    # заменяем названия на легенде
    handles, labels  =  ax.get_legend_handles_labels()
    ax.legend(handles, ['ДИ', f'{metric_name} аномалии', f'{metric_name}'])
    plt.tick_params(axis='x', rotation=20, size=17)
    plt.tick_params(axis='y', size=17)
    plt.tight_layout()
    # созданяем график в объект
    plot_object = io.BytesIO()
    plt.savefig(plot_object)
    plot_object.seek(0)
    plot_object.name = f'{metric}_plot.png'
    plt.close()

    return plot_object



def send_msg(bot_token: str, chat_id: str, message: str):
    '''
    Функция для отправки соообщения\n
    Параметры:\n
    bot_token: str\n
        Токен бота в telegram\n
    chat_id: str\n
        ID чата\n
    message: str\n
        Текст сообщения\n
    '''
    return requests.post(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}')



def send_photo(bot_token: str, chat_id: str, photo):
    '''
    Функция для отправки графиков\n
    Параметры:\n
    bot_token: str\n
        Токен бота в telegram\n
    chat_id: str\n
        ID чата\n
    photo\n
        График для отправки\n
    '''
    photo.seek(0)
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto?chat_id={chat_id}'
    requests.post(url, files={'photo': photo})


query_feed = '''
select  toStartOfFifteenMinutes(time) as time_period,
        count(distinct user_id) as active_users,
        countIf(user_id, action = 'view') as views,
        countIf(user_id, action = 'like') as likes,
        countIf(user_id, action = 'like') / countIf(action = 'view') as ctr
from    simulator_20231113.feed_actions
group by toStartOfFifteenMinutes(time)
having toStartOfFifteenMinutes(time) >= toStartOfDay(today()) - interval '15 day' and toStartOfFifteenMinutes(time) < toStartOfFifteenMinutes(now())
order by toStartOfFifteenMinutes(time)
'''

query_messages = '''
select  toStartOfFifteenMinutes(time) as time_period,
        count(distinct user_id) as active_users,
        count(user_id) as messages
from    simulator_20231113.message_actions as ma
group by toStartOfFifteenMinutes(time)
having toStartOfFifteenMinutes(time) >= toStartOfDay(today()) - interval '15 day' and toStartOfFifteenMinutes(time) < toStartOfFifteenMinutes(now())
order by toStartOfFifteenMinutes(time)
'''


# создаем list с запросами
querys_lst = [query_feed, query_messages]
# создаем list с форматами данных
querys_dtypes =[
    {
        'active_users': 'int32',
        'views': 'int64',
        'likes': 'int64'},
    {
        'active_users': 'int64',
        'messages': 'int64'}]
# list с названиями запросов
querys_names_lst = ['feed', 'messages']
# list с метриками
metrics_lst = [
    ['active_users', 'views', 'likes', 'ctr'],
    ['active_users', 'messages']
]
# list с названиеями метрик для отображения на графиках у вставки в сообщения
metric_names_lst = [
    ['Пользователи', 'Просмотры', 'Лайки', 'CTR'],
    ['Пользователи', 'Сообщения']
]
# dict с параметрами для функции find_anomalies
# параметры подобраны для каждой метрики на основе изучения данных за последние 2 месяца
anomalies_dict = {
    'feed': {
        # days, periods, rolling_n, days_alpha, periods_alpha, rolling_func
        'active_users': [5, 5, 10, 4, 3.2, 'median'],
        'views': [3, 5, 4, 2.5, 3.5, 'median'],
        'likes': [3, 5, 4, 2.5, 2.4, 'median'],
        'ctr': [3, 5, 5, 2, 3.5, 'median']
    },
    'messages':{
        # days, periods, rolling_n, days_alpha, periods_alpha, rolling_func
        'active_users': [5, 5, 5, 2.5, 2.5, 'median'],
        'messages': [5, 5, 5, 2.5, 2.6, 'median']
    }
}


@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False, concurrency=4)
def dag_send_alerts_report_chernovol():
    @task
    def make_anomalies_report(df: pd.core.frame.DataFrame, metrics: list, metric_names: list, anomalies_dict: dict, query_name: str, dashboard: str):
        '''
        Функция для создания отчета об аномальных значениях.\n
        Параметры:\n
        df: pd.core.frame.DataFrame\n
            Датафрейм с рассчитанными метриками
        metrics: list\n
            list с метриками, рассчитыными в запросах\n
        metric_names: list\n
            list с названиеми метрик для подписей на графиках и указания в сообщении\n
        anomalies_dict: dict\n
            dict с указанием параметров для поиска аномалий для функции find_anomalies\n
        query_name: list\n
            Название запроса\n
        Пример:\n

            metrics = ['active_users', 'likes']

            metric_names_lst = ['Активные пользователи', 'Лайки']

            anomalies_dict = {
                'feed': {
                    'active_users': [5, 5, 10, 3, 3, 'median'],
                    'likes': [3, 5, 15, 2.4, 2.4, 'median']}

            query_name = 'feed'
        '''
        messages =list()
        photos = list()
        alert = False

        for m, n in zip(metrics, metric_names):
            dv = anomalies_dict[m]
            df_an = find_anomalies(df, m, dv[0], dv[1], dv[2], dv[3], dv[4], dv[5])
            if df_an[f'anomaly_{m}'].iat[-1]:
                alert = True
                print(f'We got anomaly in {query_name} - {n}')
                current_value = df_an[f'{m}'].iat[-1]
                prev_value = df_an[f'{m}'].iat[-2]
                message = make_message(n, current_value, prev_value, query_name, df_an['time_period'].iat[-1], dashboard)
                df_plt = df_an.copy()[df_an['time_period'] > df_an['time_period'].max() - datetime.timedelta(days=2)]
                photo = make_graph(df_plt, m, n, query_name)
                messages.append(message)
                photos.append(photo)
            else:
                continue
        return alert, messages, photos

    @task
    def concat_reports(*reports):
        '''
        Функция для объединения отчетов.\n
        Параметры:\n
            reports\n
            Указываем отчеты полученые с помощью функции make_anomalies_report
        '''
        all_reports = list()
        for r in reports:
            if r[0]:
                all_reports.append(r)
            else:
                continue
        return all_reports

    @task
    def send_summarize(bot_token: str, chat_id: str, all_reports: list):
        '''
        Функция для отправки отчетов.\n
        Функции send_msg и send_photo объединены в одном @task, чтобы не было временного лага между отправков сообщения и графиков.\n
        Параметры:\n
        bot_token: str\n
            Токен бота в telegram\n
        chat_id: str\n
            ID чата\n
        all_reports: list\n
            Список всех отчетов, полученных с помощью функции concat_reports\n
        '''
        for r in all_reports:
            if r[0]:
                for m, p in zip(r[1], r[2]):
                    send_msg(bot_token, chat_id, m)
                    send_photo(bot_token, chat_id, p)

    # Указываем токен бота, id чата, ссылку на дэшборд с метриками
    bot_token = os.environ.get('telegram_bot_token')
    chat_id = '-938659451'
    dashboard = 'https://superset.lab.karpov.courses/superset/dashboard/****/'

    # Создаем отчет по ленте новостей
    result_feed_task = make_anomalies_report(
        extract_from_db(querys_lst[0], querys_dtypes[0]),
        metrics_lst[0],
        metric_names_lst[0],
        anomalies_dict[querys_names_lst[0]],
        querys_names_lst[0],
        dashboard)

    # Создаем отчет по сообщениям
    result_messages_task = make_anomalies_report(
        extract_from_db(querys_lst[1], querys_dtypes[1]),
        metrics_lst[1],
        metric_names_lst[1],
        anomalies_dict[querys_names_lst[1]],
        querys_names_lst[1],
        dashboard)

    # Объединяем отчеты
    all_reports_task = concat_reports(result_feed_task, result_messages_task)

    # Отправка отчетов
    send_report_task = send_summarize(bot_token, chat_id, all_reports_task)

    # Задаем порядок выполнения tasks
    send_report_task.set_upstream(all_reports_task)
    all_reports_task.set_upstream(result_messages_task)
    all_reports_task.set_upstream(result_feed_task)


dag_send_alerts_report_chernovol = dag_send_alerts_report_chernovol()