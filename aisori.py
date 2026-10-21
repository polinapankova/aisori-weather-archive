"""Модуль для работы с архивами наблюдений АИСОРИ (meteo.ru).

Выгрузка АИСОРИ приходит в формате фиксированной ширины: без
разделителей, без заголовка, с пропусками в виде пробелов и с кодовыми
полями вместо привычных величин. Модуль переводит ее в удобный вид
и считает средние, норму, аномалии и тренд.

Данные хранятся в словаре: ключ -- имя колонки, значение -- список
значений по всем наблюдениям. Пропущенному значению соответствует None.

Модуль состоит из четырех частей:
    1. Описание полей -- какой ширины колонки и как они называются.
    2. Чтение архива -- загрузка в словарь и сохранение в CSV.
    3. Расшифровка кодов -- видимость, облачность, направление ветра.
    4. Расчеты -- средние по суткам и годам, норма, аномалии, тренд.

Пример:
    import aisori

    table = aisori.read_fields('data/sample/fld27612.txt')
    data = aisori.read_archive('data/sample/wr27612_1966.txt', table,
                               ['year', 'month', 'day', 'air_temp'])
    days = aisori.day_means(aisori.group_by_day(data, 'air_temp'))
    print(aisori.year_means(days))
"""

import numpy as np


# ====================================================================
# ЧАСТЬ 1. ОПИСАНИЕ ПОЛЕЙ ВЫГРУЗКИ
# ====================================================================


# Кодировка, в которой АИСОРИ отдает текстовые файлы.
ENCODING = 'cp1251'

# Словарь соответствия: русское название поля -> короткое имя колонки.
# Названия записаны в нижнем регистре и с одиночными пробелами, потому
# что функция read_fields приводит их к такому виду.
NAMES = {
    'синоптический индекс станции': 'station',
    'год по гринвичу': 'year_utc',
    'месяц по гринвичу': 'month_utc',
    'день по гринвичу': 'day_utc',
    'срок по гринвичу': 'hour_utc',
    'год источника (местный)': 'year',
    'месяц источника (местный)': 'month',
    'день источника (местный)': 'day',
    'срок источника (местный)': 'hour',
    'номер срока в сутках по пдзв': 'term',
    'время местное': 'local_time',
    'номер часового пояса': 'time_zone',
    'начало метеорологических суток по пдзв': 'day_start',
    'горизонтальная дальность видимости': 'visibility_code',
    'общее количество облачности': 'cloud_code',
    'количество облачности нижнего яруса': 'cloud_low_code',
    'форма облаков верхнего яруса': 'cloud_high_form',
    'форма облаков среднего яруса': 'cloud_mid_form',
    'форма облаков вертикального развития': 'cloud_vertical_form',
    'слоистые и слоисто-кучевые облака': 'cloud_st_sc',
    'слоисто-дожд,разорванно-дождевые облака': 'cloud_ns',
    'высота нижней границы облачности': 'cloud_height_code',
    'признак наличия облачности ниже уровня станции': 'cloud_below',
    'погода между сроками': 'weather_past',
    'погода в срок наблюдения': 'weather_now',
    'направление ветра': 'wind_direction',
    'средняя скорость ветра': 'wind_speed',
    'максимальная скорость ветра': 'wind_speed_max',
    'сумма осадков': 'precipitation',
    'температура поверхности почвы': 'soil_temp',
    'температура пов. почвы по мин. терм-ру': 'soil_temp_min_t',
    'мин. температура пов-сти почвы между сроками': 'soil_temp_min',
    'макс. температура пов-сти почвы между сроками': 'soil_temp_max',
    'температура пов-сти почвы по макс. терм-ру п/встр.': 'soil_temp_max_t',
    'температура воздуха по сухому терм-ру': 'air_temp',
    'темп.воздуха по смоченному терм-ру': 'air_temp_wet',
    'температура воздуха по мин. терм-ру': 'air_temp_min_t',
    'мин.температура воздуха между сроками': 'air_temp_min',
    'макс. темперура воздуха между сроками': 'air_temp_max',
    'темпера воздуха по макс. терм-ру после встрях.': 'air_temp_max_t',
    'парциальное давление водяного пара': 'vapour_pressure',
    'относительная влажность воздуха': 'humidity',
    'дефицит насыщения водяного пара': 'saturation_deficit',
    'температура точки росы': 'dew_point',
    'атмосферное давление на уровне станции': 'pressure',
    'атмосферное давление на уровне моря': 'pressure_sea',
    'характеристика барической тенденции': 'pressure_change_type',
    'величина барической тенденции': 'pressure_change',
}


def read_fields(path):
    """Читает описание полей и считает границы каждого поля в строке.

    В записи поля идут подряд и разделены одним пробелом, поэтому
    начало следующего поля равно концу предыдущего плюс один знак.

    Аргументы:
        path: путь к файлу описания полей fld*.txt.

    Возвращает:
        Список кортежей (имя, начало, конец). Имя берется из словаря
        NAMES, а если названия там нет, получается вида 'f07'.
    """
    fields = []
    start = 0
    with open(path, encoding=ENCODING) as data_file:
        for line in data_file:
            parts = line.split()
            # В файле бывают пустые строки и примечания, пропускаем их.
            if len(parts) < 4:
                continue
            if not parts[0].isdigit() or not parts[1].isdigit():
                continue
            # Ширина записана как '5' или как '5,1' - всего знаков
            # и знаков после запятой. Нам нужна только полная ширина.
            width_text = parts[2].split(',')[0]
            if not width_text.isdigit():
                continue
            width = int(width_text)
            # Объединение через пробел убирает двойные пробелы,
            # которые встречаются в названиях.
            title = ' '.join(parts[3:]).lower()
            name = NAMES.get(title, 'f' + parts[0])
            fields.append((name, start, start + width))
            start = start + width + 1
    return fields


def read_titles(path):
    """Собирает справочник: короткое имя колонки -> русское название.

    Аргументы:
        path: путь к файлу описания полей fld*.txt.

    Возвращает:
        Словарь, пригодный для вывода расшифровки колонок.
    """
    titles = {}
    with open(path, encoding=ENCODING) as data_file:
        for line in data_file:
            parts = line.split()
            if len(parts) < 4:
                continue
            if not parts[0].isdigit() or not parts[1].isdigit():
                continue
            title = ' '.join(parts[3:])
            name = NAMES.get(title.lower(), 'f' + parts[0])
            titles[name] = title
    return titles


def line_length(fields):
    """Считает ожидаемую длину записи без знаков переноса строки.

    Если фактическая длина строки другая, значит описание полей
    не подходит к файлу данных.

    Аргументы:
        fields: список полей из read_fields.

    Возвращает:
        Число знаков в записи.
    """
    if len(fields) == 0:
        return 0
    # Конец последнего поля и есть длина записи: разделители уже
    # учтены при подсчете границ в read_fields.
    return fields[-1][2]


def read_stations(path):
    """Читает список станций выгрузки.

    Аргументы:
        path: путь к файлу statlist*.txt.

    Возвращает:
        Словарь: синоптический индекс станции -> название.
    """
    stations = {}
    with open(path, encoding=ENCODING) as data_file:
        for line in data_file:
            parts = line.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            stations[int(parts[0])] = ' '.join(parts[1:])
    return stations


def station_name(path):
    """Возвращает название станции с наименьшим индексом из списка.

    В большинстве выгрузок станция одна, и функция избавляет
    от того, чтобы доставать название из словаря вручную.

    Аргументы:
        path: путь к файлу statlist*.txt.

    Возвращает:
        Название станции или None, если список пуст.
    """
    stations = read_stations(path)
    if len(stations) == 0:
        return None
    return stations[min(stations.keys())]

# ====================================================================
# ЧАСТЬ 2. ЧТЕНИЕ АРХИВА И СОХРАНЕНИЕ
# ====================================================================


def to_number(text):
    """Переводит кусок строки в число.

    Аргументы:
        text: кусок строки записи.

    Возвращает:
        Целое число, если в тексте нет точки, дробное число, если точка
        есть, и None, если поле пустое.

    Примеры:
        >>> to_number('  27612')
        27612
        >>> to_number('  -3.0')
        -3.0
        >>> print(to_number('     '))
        None
    """
    text = text.strip()
    if text == '':
        return None
    if '.' in text:
        return float(text)
    return int(text)


def read_archive(path, fields, names=None):
    """Читает файл наблюдений в словарь списков.

    Аргументы:
        path: путь к файлу данных wr*.txt.
        fields: список полей из read_fields.
        names: список нужных колонок. Если None, читаются все колонки.
            Чтение только нужных колонок заметно ускоряет работу
            на больших архивах.

    Возвращает:
        Словарь: имя колонки -> список значений.
    """
    chosen = []
    data = {}
    for name, start, end in fields:
        if names is None or name in names:
            chosen.append((name, start, end))
            data[name] = []
    with open(path) as data_file:
        for line in data_file:
            for name, start, end in chosen:
                data[name].append(to_number(line[start:end]))
    return data


def check_line_length(path, fields):
    """Сверяет длину первой записи с описанием полей.

    Файлы АИСОРИ приходят с переносом строки из двух знаков, поэтому
    длину надо считать после удаления знаков переноса. Иначе запись
    кажется на один знак длиннее, чем есть данных.

    Аргументы:
        path: путь к файлу данных wr*.txt.
        fields: список полей из read_fields.

    Возвращает:
        True, если длина совпала, и False, если нет.
    """
    with open(path) as data_file:
        first = data_file.readline()
    first = first.rstrip('\r\n')
    return len(first) == line_length(fields)


def count_rows(data):
    """Считает число наблюдений в словаре данных.

    Аргументы:
        data: словарь с данными.

    Возвращает:
        Число наблюдений, 0 для пустого словаря.
    """
    for name in data:
        return len(data[name])
    return 0


def stations_in(data):
    """Перечисляет станции, которые встречаются в данных.

    В одной выгрузке может оказаться несколько станций. Перед расчетом
    средних это нужно проверить: наблюдения разных станций за одну дату
    смешаются, и среднее окажется бессмысленным.

    Аргументы:
        data: словарь с данными, в котором есть колонка 'station'.

    Возвращает:
        Список индексов станций в порядке появления.
    """
    found = []
    for value in data['station']:
        if value not in found:
            found.append(value)
    return found


def select_station(data, index):
    """Оставляет наблюдения только одной станции.

    Колонку 'station' нужно прочитать вместе с остальными, иначе
    функция не найдет, по чему отбирать.

    Аргументы:
        data: словарь с данными.
        index: синоптический индекс станции, например 27612.

    Возвращает:
        Новый словарь того же вида, но только с нужной станцией.
    """
    result = {}
    for name in data:
        result[name] = []
    for number in range(count_rows(data)):
        if data['station'][number] == index:
            for name in data:
                result[name].append(data[name][number])
    return result


def save_csv(data, path):
    """Сохраняет словарь данных в CSV с запятой и кодировкой UTF-8.

    Это главная цель модуля: превратить формат фиксированной ширины
    в файл, который без настройки открывают Excel, QGIS и pandas.

    Аргументы:
        data: словарь с данными.
        path: путь к файлу результата.
    """
    names = list(data.keys())
    rows = count_rows(data)
    with open(path, 'w', encoding='utf-8') as out_file:
        out_file.write(','.join(names) + '\n')
        for number in range(rows):
            values = []
            for name in names:
                value = data[name][number]
                if value is None:
                    values.append('')
                else:
                    values.append(str(value))
            out_file.write(','.join(values) + '\n')


def save_table(table, path, first_title, second_title):
    """Сохраняет словарь «ключ -> число» в CSV из двух колонок.

    Подходит для суточных, месячных и годовых средних.

    Аргументы:
        table: словарь, например {1966: 5.27, 1967: 5.35}.
        path: путь к файлу результата.
        first_title: заголовок первой колонки.
        second_title: заголовок второй колонки.
    """
    keys = sorted(table.keys())
    with open(path, 'w', encoding='utf-8') as out_file:
        out_file.write(first_title + ',' + second_title + '\n')
        for key in keys:
            # Ключом может быть год (число) или дата (кортеж).
            if type(key) is tuple:
                key_text = '-'.join([str(part) for part in key])
            else:
                key_text = str(key)
            out_file.write(key_text + ',' + str(table[key]) + '\n')

# ====================================================================
# ЧАСТЬ 3. РАСШИФРОВКА КОДОВЫХ ПОЛЕЙ
# ====================================================================


# Код направления ветра, которым отмечено отсутствие данных.
WIND_MISSING = 999

# Код направления ветра, которым отмечен штиль.
WIND_CALM = 0

# Наибольший балл общего количества облачности. Значения выше
# относятся к служебным кодам, смысл которых нужно уточнять.
CLOUD_MAX = 10

# Служебные коды видимости из таблицы ВМО 4377 и их значение в км.
# Код 90 означает «менее 0,05 км», код 99 -- «50 км и более»,
# для них взяты граничные значения.
VISIBILITY_SPECIAL = {
    90: 0.05,
    91: 0.05,
    92: 0.2,
    93: 0.5,
    94: 1.0,
    95: 2.0,
    96: 4.0,
    97: 10.0,
    98: 20.0,
    99: 50.0,
}

# Названия румбов для розы ветров: от севера по часовой стрелке.
COMPASS = ['С', 'СВ', 'В', 'ЮВ', 'Ю', 'ЮЗ', 'З', 'СЗ']


def visibility_km(code):
    """Переводит код горизонтальной видимости в километры.

    Используется таблица ВМО 4377:
    0 -- менее 0,1 км; от 1 до 50 -- десятые доли километра;
    от 56 до 80 -- километры как код минус 50; от 81 до 88 --
    от 35 до 70 км с шагом 5 км; 89 -- более 70 км; от 90 до 99 --
    служебные коды из словаря VISIBILITY_SPECIAL. Коды от 51 до 55
    в таблице не используются.

    В пользу этой таблицы говорит то, что в архиве станции
    Москва, ВДНХ нет ни одного кода из диапазона 51-55. Но сверить
    ее с документацией вашей выгрузки все равно стоит.

    Аргументы:
        code: код видимости или None.

    Возвращает:
        Видимость в километрах или None.

    Примеры:
        >>> visibility_km(25)
        2.5
        >>> visibility_km(95)
        2.0
        >>> print(visibility_km(52))
        None
    """
    if code is None:
        return None
    if code == 0:
        return 0.05
    if code >= 1 and code <= 50:
        return code / 10
    if code >= 56 and code <= 80:
        return float(code - 50)
    if code >= 81 and code <= 88:
        return float(30 + (code - 80) * 5)
    if code == 89:
        return 75.0
    if code in VISIBILITY_SPECIAL:
        return VISIBILITY_SPECIAL[code]
    return None


def cloud_fraction(code):
    """Переводит количество облачности из баллов в долю неба.

    Баллы от 0 до 10 делятся на 10. Значения выше 10 -- служебные
    коды, их смысл нужно взять из документации АИСОРИ, поэтому
    функция возвращает для них None.

    В данных этого проекта код 12 встречается примерно в 6
    процентах записей, то есть молча отбрасывать его нельзя:
    сначала надо выяснить, что он означает.

    Аргументы:
        code: количество облачности в баллах или None.

    Возвращает:
        Долю закрытого неба от 0 до 1 или None.

    Примеры:
        >>> cloud_fraction(5)
        0.5
        >>> print(cloud_fraction(12))
        None
    """
    if code is None:
        return None
    if code < 0 or code > CLOUD_MAX:
        return None
    return code / CLOUD_MAX


def wind_degrees(code):
    """Оставляет только осмысленное направление ветра.

    Код 999 означает отсутствие данных, код 0 -- штиль, при котором
    направление не определено. Для обоих возвращается None.

    Аргументы:
        code: направление ветра в градусах или None.

    Возвращает:
        Направление от 1 до 360 градусов или None.

    Примеры:
        >>> wind_degrees(90)
        90
        >>> print(wind_degrees(999))
        None
    """
    if code is None:
        return None
    if code == WIND_CALM or code == WIND_MISSING:
        return None
    return code


def calm_share(codes):
    """Считает долю наблюдений со штилем.

    Аргументы:
        codes: список кодов направления ветра.

    Возвращает:
        Долю от 0 до 1.
    """
    if len(codes) == 0:
        return 0.0
    calm = 0
    for code in codes:
        if code == WIND_CALM:
            calm = calm + 1
    return calm / len(codes)


def decode_column(values, rule):
    """Применяет расшифровку ко всем значениям колонки.

    Аргументы:
        values: список кодов.
        rule: функция расшифровки, например visibility_km.

    Возвращает:
        Новый список расшифрованных значений.
    """
    result = []
    for value in values:
        result.append(rule(value))
    return result


def wind_rose(directions):
    """Считает повторяемость направлений ветра по восьми румбам.

    Границы румбов сдвинуты на половину сектора, чтобы северный
    румб был центрирован на нуле градусов.

    Аргументы:
        directions: список кодов направления ветра.

    Возвращает:
        Словарь: название румба -> доля наблюдений от 0 до 1.
    """
    counts = {}
    for name in COMPASS:
        counts[name] = 0
    total = 0
    width = 360 / len(COMPASS)
    for code in directions:
        degrees = wind_degrees(code)
        if degrees is None:
            continue
        number = int((degrees + width / 2) % 360 // width)
        counts[COMPASS[number]] = counts[COMPASS[number]] + 1
        total = total + 1
    if total == 0:
        return counts
    result = {}
    for name in COMPASS:
        result[name] = counts[name] / total
    return result

# ====================================================================
# ЧАСТЬ 4. СРЕДНИЕ, НОРМА, АНОМАЛИИ, ТРЕНД
# ====================================================================


# Сроков в сутках в срочной выгрузке: наблюдения каждые три часа.
TERMS_PER_DAY = 8

# Названия месяцев в родительном падеже: «1 января», «15 марта».
MONTH_NAMES = {
    1: 'января',
    2: 'февраля',
    3: 'марта',
    4: 'апреля',
    5: 'мая',
    6: 'июня',
    7: 'июля',
    8: 'августа',
    9: 'сентября',
    10: 'октября',
    11: 'ноября',
    12: 'декабря',
}


def month_name(number):
    """Возвращает название месяца в родительном падеже.

    Аргументы:
        number: номер месяца от 1 до 12.

    Возвращает:
        Название месяца или None, если номер неверный.

    Примеры:
        >>> month_name(1)
        'января'
        >>> print(month_name(13))
        None
    """
    if number in MONTH_NAMES:
        return MONTH_NAMES[number]
    return None


def first_date(days):
    """Возвращает самую раннюю дату словаря с ключами-датами.

    Даты хранятся кортежами (год, месяц, день), и функция min
    сравнивает их по порядку: сначала годы, потом месяцы, потом дни.
    Поэтому ответ не зависит от порядка записей в файле.

    Аргументы:
        days: словарь из group_by_day или day_means.

    Возвращает:
        Кортеж (год, месяц, день) или None, если словарь пуст.

    Примеры:
        >>> first_date({(1966, 3, 2): 1.0, (1966, 1, 5): 2.0})
        (1966, 1, 5)
    """
    if len(days) == 0:
        return None
    return min(days.keys())


def mean(values):
    """Считает среднее по списку, пропуская отсутствующие значения.

    Аргументы:
        values: список чисел, в котором могут быть None.

    Возвращает:
        Среднее значение или None, если чисел не нашлось.

    Примеры:
        >>> mean([1.0, 2.0, None, 3.0])
        2.0
        >>> print(mean([None]))
        None
    """
    total = 0.0
    count = 0
    for value in values:
        if value is not None:
            total = total + value
            count = count + 1
    if count == 0:
        return None
    return total / count


def missing_share(values):
    """Считает долю пропущенных значений в списке.

    Аргументы:
        values: список значений, в котором могут быть None.

    Возвращает:
        Долю от 0 до 1. Для пустого списка возвращает 0.
    """
    if len(values) == 0:
        return 0.0
    missing = 0
    for value in values:
        if value is None:
            missing = missing + 1
    return missing / len(values)


def group_by_day(data, name):
    """Собирает значения колонки по суткам.

    Используется местная дата, а не дата по Гринвичу: иначе сутки
    сдвигаются на несколько часов и средние получаются смещенными.

    Аргументы:
        data: словарь с данными.
        name: имя колонки, например 'air_temp'.

    Возвращает:
        Словарь: (год, месяц, день) -> список значений за эти сутки.
    """
    groups = {}
    years = data['year']
    months = data['month']
    days = data['day']
    values = data[name]
    for number in range(len(values)):
        key = (years[number], months[number], days[number])
        if key not in groups:
            groups[key] = []
        groups[key].append(values[number])
    return groups


def day_means(groups, min_terms=6):
    """Считает средние по суткам и отбрасывает неполные сутки.

    Аргументы:
        groups: словарь из group_by_day.
        min_terms: наименьшее число сроков в сутках. По умолчанию
            требуется 6 сроков из 8.

    Возвращает:
        Словарь: (год, месяц, день) -> среднее за сутки.
    """
    result = {}
    for key in groups:
        values = groups[key]
        present = 0
        for value in values:
            if value is not None:
                present = present + 1
        if present >= min_terms:
            result[key] = mean(values)
    return result


def month_means(days, min_days=25):
    """Считает средние по месяцам из суточных средних.

    Аргументы:
        days: словарь суточных средних из day_means.
        min_days: наименьшее число суток в месяце.

    Возвращает:
        Словарь: (год, месяц) -> среднее за месяц.
    """
    groups = {}
    for key in days:
        year, month, day = key
        month_key = (year, month)
        if month_key not in groups:
            groups[month_key] = []
        groups[month_key].append(days[key])
    result = {}
    for key in groups:
        if len(groups[key]) >= min_days:
            result[key] = mean(groups[key])
    return result


def year_means(days, min_days=350):
    """Считает средние по годам из суточных средних.

    Аргументы:
        days: словарь суточных средних из day_means.
        min_days: наименьшее число суток в году. Значение по умолчанию
            допускает до двух недель пропусков.

    Возвращает:
        Словарь: год -> среднее за год.
    """
    groups = {}
    for key in days:
        year = key[0]
        if year not in groups:
            groups[year] = []
        groups[year].append(days[key])
    result = {}
    for year in groups:
        if len(groups[year]) >= min_days:
            result[year] = mean(groups[year])
    return result


def norm_by_month(months, first_year=1991, last_year=2020, min_years=20):
    """Считает климатическую норму по месяцам за базовый период.

    Всемирная метеорологическая организация рекомендует считать
    норму по тридцатилетию; действующий период -- 1991-2020 годы,
    он и взят по умолчанию. Для другого периода передайте свои
    границы. Если архив покрывает период не целиком, норма
    считается по той части, которая есть, и это нужно оговорить
    в выводах.

    Аргументы:
        months: словарь месячных средних из month_means.
        first_year: первый год базового периода.
        last_year: последний год базового периода.
        min_years: наименьшее число лет для каждого месяца.

    Возвращает:
        Словарь: номер месяца -> норма.
    """
    groups = {}
    for key in months:
        year, month = key
        if year >= first_year and year <= last_year:
            if month not in groups:
                groups[month] = []
            groups[month].append(months[key])
    result = {}
    for month in groups:
        if len(groups[month]) >= min_years:
            result[month] = mean(groups[month])
    return result


def used_years(months, first_year=1991, last_year=2020):
    """Показывает, по каким годам фактически посчитана норма.

    Аргументы:
        months: словарь месячных средних.
        first_year: первый год базового периода.
        last_year: последний год базового периода.

    Возвращает:
        Кортеж (первый год, последний год, число лет). Если данных
        в базовом периоде нет, возвращает (None, None, 0).
    """
    years = []
    for key in months:
        year = key[0]
        if year >= first_year and year <= last_year:
            if year not in years:
                years.append(year)
    if len(years) == 0:
        return (None, None, 0)
    return (min(years), max(years), len(years))


def year_anomalies(years, norm):
    """Считает отклонения годовых средних от среднего по норме.

    Аргументы:
        years: словарь годовых средних.
        norm: словарь нормы по месяцам.

    Возвращает:
        Словарь: год -> отклонение от средней годовой нормы.
    """
    year_norm = mean(list(norm.values()))
    result = {}
    for year in years:
        result[year] = years[year] - year_norm
    return result


def trend(table):
    """Оценивает линейный тренд методом наименьших квадратов.

    Прямая ищется руками по формуле наклона: сумма произведений
    отклонений от средних, деленная на сумму квадратов отклонений
    по оси лет.

    Аргументы:
        table: словарь год -> значение.

    Возвращает:
        Кортеж (наклон за десятилетие, свободный член). Если точек
        меньше двух, возвращает (None, None).
    """
    if len(table) < 2:
        return (None, None)
    years = np.array(sorted(table.keys()), dtype=float)
    values = np.array([table[int(year)] for year in years], dtype=float)
    year_mean = years.mean()
    value_mean = values.mean()
    top = ((years - year_mean) * (values - value_mean)).sum()
    bottom = ((years - year_mean) ** 2).sum()
    slope = top / bottom
    intercept = value_mean - slope * year_mean
    return (slope * 10, intercept)


def extremes(table, count=5):
    """Находит наибольшие и наименьшие значения словаря.

    Аргументы:
        table: словарь ключ -> значение.
        count: сколько значений показать с каждого края.

    Возвращает:
        Кортеж из двух списков пар (ключ, значение): сначала
        наибольшие по убыванию, затем наименьшие по возрастанию.
    """
    pairs = []
    for key in table:
        pairs.append((table[key], key))
    pairs.sort()
    smallest = []
    for value, key in pairs[:count]:
        smallest.append((key, value))
    largest = []
    for value, key in pairs[-count:][::-1]:
        largest.append((key, value))
    return (largest, smallest)
