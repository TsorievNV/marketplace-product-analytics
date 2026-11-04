from pathlib import Path
import json
import math
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DECISION_LABELS = {
    'SIMULATED GO': 'Можно запускать в рамках симуляции',
    'SIMULATED HOLD': 'Пока не запускать в рамках симуляции',
}


def generate(plan):
    """Одна поисковая сессия на пользователя; упрощённая симуляция с полным логом."""
    rng = np.random.default_rng(plan['seed'])
    n = plan['users']
    assigned = pd.Timestamp('2026-10-01') + pd.to_timedelta(
        rng.integers(0, plan['enrollment_days'] * 86400, n), unit='s')
    a = pd.DataFrame({
        'user_id': np.arange(n), 'variant': np.where(rng.random(n) < .5, 'A', 'B'),
        'device': rng.choice(['mobile', 'desktop'], n, p=[.72, .28]),
        'category': rng.choice(['electronics', 'home', 'hobby'], n),
        'prior_sessions': rng.poisson(2, n), 'assigned_at': assigned,
        'window_end': assigned + pd.Timedelta(days=plan['followup_days'])})
    # Характеристики известны до назначения. Эффект изменения задан генератором.
    p = (.14 + .04 * (a.device == 'mobile') + .025 * np.minimum(a.prior_sessions, 5)
         + .025 * (a.category == 'electronics') + .018 * (a.variant == 'B'))
    viewed = rng.random(n) < .66
    contacted = viewed & (rng.random(n) < np.asarray(p) / .66)
    complained = rng.random(n) < (.008 + .0005 * (a.variant == 'B'))
    rows = []
    for name, mask, seconds in [('search', np.ones(n, bool), 60),
                                ('listing_view', viewed, 120),
                                ('contact', contacted, 240),
                                ('complaint', complained, 360)]:
        part = a.loc[mask, ['user_id', 'assigned_at']].copy()
        part['session_id'] = part.user_id.map(lambda x: f's{x}')
        part['listing_id'] = (part.user_id % 1000).astype(str)
        part['event_name'] = name
        part['event_time'] = part.assigned_at + pd.to_timedelta(seconds, unit='s')
        part['received_at'] = part.event_time + pd.Timedelta(seconds=2)
        rows.append(part.drop(columns='assigned_at'))
    e = pd.concat(rows, ignore_index=True)
    e['event_id'] = np.arange(len(e)).astype(str)
    # Добавляем дефекты поверх корректного лога, сохраняя исходные действия и пользователей.
    duplicates = e.sample(frac=.025, random_state=17).copy()
    duplicates['received_at'] += pd.Timedelta(seconds=30)
    bad = e.head(45).copy()
    bad['event_id'] = [f'bad{i}' for i in range(len(bad))]
    bad.loc[bad.index[:15], 'user_id'] = n + 1
    bad.loc[bad.index[15:30], 'event_time'] = pd.NaT
    bad.loc[bad.index[30:], 'event_name'] = 'listing_view'
    bad.loc[bad.index[30:], 'listing_id'] = None
    raw = pd.concat([e, duplicates, bad], ignore_index=True)
    return a, raw


def clean_events(assignments, raw):
    """Изолируем некорректные строки, сохраняя всех назначенных пользователей в знаменателе."""
    x = raw.copy()
    x['event_time'] = pd.to_datetime(x.event_time, errors='coerce')
    x['received_at'] = pd.to_datetime(x.received_at, errors='coerce')
    payload = ['user_id', 'session_id', 'listing_id', 'event_name', 'event_time']
    conflict = x.groupby('event_id')[payload].nunique(dropna=False).gt(1).any(axis=1)
    if conflict.any():
        raise ValueError('Разные данные у событий с одним event_id')
    dup_count = int(x.duplicated('event_id').sum())
    x = x.sort_values('received_at').drop_duplicates('event_id')
    known = x.user_id.isin(assignments.user_id)
    valid_time = x.event_time.notna() & x.received_at.notna() & (x.received_at >= x.event_time)
    valid_type = x.event_name.isin(['search', 'listing_view', 'contact', 'complaint'])
    valid_ids = x.session_id.notna() & x.event_id.notna()
    valid_listing = ~x.event_name.isin(['listing_view', 'contact']) | x.listing_id.notna()
    valid = known & valid_time & valid_type & valid_ids & valid_listing
    audit = {'raw_rows': len(raw), 'duplicate_rows': dup_count,
             'unknown_user_rows': int((~known).sum()),
             'invalid_timestamp_rows': int((~valid_time).sum()),
             'missing_listing_rows': int((~valid_listing).sum()),
             'quarantined_rows': int((~valid).sum()), 'clean_rows': int(valid.sum()),
             'note': 'Строка может нарушать несколько правил. Потери неотправленных событий здесь не проверяются.'}
    return x.loc[valid].copy(), x.loc[~valid].copy(), audit


def build_mart(a, e, database=':memory:'):
    if not a.user_id.is_unique or a.user_id.isna().any():
        raise ValueError('user_id в таблице назначений должен быть уникальным и заполненным')
    if not a.variant.isin(['A', 'B']).all():
        raise ValueError('Неизвестная группа эксперимента')
    with sqlite3.connect(str(database)) as con:
        # Единый формат ISO позволяет сравнивать время в строковом представлении.
        for name, frame in [('assignments', a), ('events', e)]:
            frame = frame.copy()
            for col in ['assigned_at', 'window_end', 'event_time', 'received_at']:
                if col in frame:
                    frame[col] = pd.to_datetime(frame[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            frame.to_sql(name, con, index=False, if_exists='replace')
        con.execute('CREATE INDEX IF NOT EXISTS ix_events_user_time ON events(user_id,event_time)')
        return pd.read_sql_query((ROOT / 'sql/user_mart.sql').read_text(encoding='utf-8'), con)


def required_n(p=.22, delta=.018, alpha=.05, power=.8):
    p2 = p + delta
    avg = (p + p2) / 2
    numerator = (norm.ppf(1-alpha/2)*math.sqrt(2*avg*(1-avg))
                 + norm.ppf(power)*math.sqrt(p*(1-p)+p2*(1-p2)))**2
    return math.ceil(numerator / delta**2)


def compare_rates(a_success, a_n, b_success, b_n):
    pa, pb = a_success / a_n, b_success / b_n
    diff = pb - pa
    se = math.sqrt(pa*(1-pa)/a_n + pb*(1-pb)/b_n)
    pooled = (a_success+b_success)/(a_n+b_n)
    se_null = math.sqrt(pooled*(1-pooled)*(1/a_n+1/b_n))
    if se_null == 0:
        pvalue = 1.0
    else:
        pvalue = float(2*norm.sf(abs(diff/se_null)))
    return {'a_rate': pa, 'b_rate': pb, 'difference': diff,
            'relative_lift': diff/pa if pa else None,
            'ci_low': diff-norm.ppf(.975)*se, 'ci_high': diff+norm.ppf(.975)*se,
            'upper_one_sided_95': diff+norm.ppf(.95)*se, 'pvalue': pvalue,
            'method': 'z-тест двух долей с общей оценкой дисперсии; интервал Вальда с раздельными оценками; большие выборки'}


def analyze(mart, plan):
    g = mart.groupby('variant').agg(n=('user_id','size'),
             contacts=('contacted','sum'), complaints=('complained','sum'))
    a, b = g.loc['A'], g.loc['B']
    primary = compare_rates(int(a.contacts), int(a.n), int(b.contacts), int(b.n))
    guardrail = compare_rates(int(a.complaints), int(a.n), int(b.complaints), int(b.n))
    # Дополнительная проверка редких жалоб. Решение принимается по верхней границе прироста.
    guardrail['fisher_two_sided_pvalue'] = float(fisher_exact([
        [int(a.complaints), int(a.n-a.complaints)],
        [int(b.complaints), int(b.n-b.complaints)]])[1])
    counts = np.array([a.n, b.n], dtype=float)
    srm_p = float(chi2.sf(np.sum((counts-counts.sum()/2)**2/(counts.sum()/2)), 1))
    cutoff = pd.Timestamp('2026-10-22')
    all_mature = bool((pd.to_datetime(mart.window_end) <= cutoff).all())
    primary_pass = primary['ci_low'] > 0
    guardrail_pass = guardrail['upper_one_sided_95'] < plan['guardrail_margin_absolute']
    return {'sample': g.reset_index().to_dict(orient='records'),
            'planned_n_per_arm': required_n(plan['baseline_for_planning'], plan['mde_absolute']),
            'primary': primary, 'guardrail': guardrail, 'srm_pvalue': srm_p,
            'all_users_mature': all_mature, 'primary_pass': bool(primary_pass),
            'guardrail_pass': bool(guardrail_pass),
            'decision': 'SIMULATED GO' if primary_pass and guardrail_pass
              and srm_p >= plan['srm_alpha'] and all_mature else 'SIMULATED HOLD'}


def write_charts(mart, result, reports):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    f = mart.groupby('variant')[['searched','viewed','funnel_contacted']].mean()
    for variant, color in [('A','#667085'),('B','#087e8b')]:
        axes[0].plot(['Поиск','Просмотр','Контакт'], f.loc[variant]*100,
                     marker='o', lw=2.5, label='A — контроль' if variant == 'A' else 'B — с бейджем', color=color)
    axes[0].set(ylabel='Доля назначенных пользователей, %',
                title='Последовательная воронка\nСинтетические данные')
    axes[0].legend()
    p = result['primary']
    axes[1].errorbar([p['difference']*100], [0],
                    xerr=[[(p['difference']-p['ci_low'])*100],[(p['ci_high']-p['difference'])*100]],
                    fmt='o', capsize=8, color='#087e8b', markersize=9)
    axes[1].axvline(0,color='#98a2b3',ls='--')
    axes[1].set(yticks=[], xlabel='Разница B − A, процентные пункты',
                title='Изменение доли контактов\n95% доверительный интервал')
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ',')))
    fig.tight_layout(pad=2)
    fig.savefig(reports/'overview.png', dpi=180)
    plt.close(fig)


def write_report(mart, result, audit, reports):
    p, g = result['primary'], result['guardrail']
    pct = lambda x: f'{x*100:.2f}%'.replace('.', ',')
    pp = lambda x: f'{x*100:.2f} п.п.'.replace('.', ',')
    body = f'''# Анализ эксперимента в сервисе объявлений

**{DECISION_LABELS[result['decision']]} — результат симуляции, не рекомендация для Авито.**
Все пользователи и события синтетические. Генератор намеренно задаёт эффект;
анализ проверяет методику, SQL и воспроизводимость, а не открывает реальный рост.

## Решение и доказательства

Основная метрика — доля всех назначенных пользователей с контактом за 7 дней.
Контроль: {pct(p['a_rate'])}; тест: {pct(p['b_rate'])}.
Разница: **{pp(p['difference'])}**, относительное изменение {pct(p['relative_lift'])}.
95% ДИ разницы: [{pp(p['ci_low'])}; {pp(p['ci_high'])}]; p-значение = {p['pvalue']:.6f}.

Жалобы на назначенного пользователя: {pct(g['a_rate'])} → {pct(g['b_rate'])}.
Верхняя односторонняя 95% граница прироста: {pp(g['upper_one_sided_95'])}.
Допустимый прирост задан до запуска: 0,30 п.п. Проверка ограничения: {'пройдена' if result['guardrail_pass'] else 'не пройдена'}.
Отсутствие значимости жалоб само по себе не доказывает безопасность.

Проверка соотношения размеров групп (SRM): p-значение = {result['srm_pvalue']:.4f}.
Окна наблюдения всех пользователей завершены: {'да' if result['all_users_mature'] else 'нет'}.
Расчётный объём для основной метрики: {result['planned_n_per_arm']:,} на группу
(исходная конверсия 22%, минимальный обнаруживаемый эффект 1,8 п.п., уровень значимости 5%,
мощность 80%). Это не отдельный расчёт мощности жалоб.
Если по ограничению недостаточно точности, решение — отложить запуск независимо от основной метрики.

![Воронка и эффект](overview.png)

## Качество данных

Строк получено: {audit['raw_rows']:,}; дублей: {audit['duplicate_rows']:,};
карантин после дедупликации: {audit['quarantined_rows']}; принято: {audit['clean_rows']:,}.
Проверены известность пользователя, тип события, время, обязательные идентификаторы
и различия в содержимом событий с одним event_id. Назначения из знаменателя не исключались.
Внесённые дефекты аддитивные. Проверка не обнаруживает события, которые вообще не были отправлены.

## Воронка и продуктовая интерпретация

Воронка требует порядка «поиск → просмотр → контакт» в одной сессии и по одному объявлению.
Основная метрика отдельно считает любой корректный контакт в окне пользователя.
Знаменатель — назначенные пользователи, а не увидевшие бейдж или связавшиеся с продавцом.
Контакт — прокси полезного действия, не сделка и не выручка. Нет данных о продаже,
ответе продавца и удовлетворённости. Нельзя пересчитывать эффект в деньги.

## Что изменится в реальном эксперименте

Нужны серверное событие контакта, мониторинг потерь доставки, устойчивое назначение
между устройствами, аудит показа бейджа, исключение ботов по правилам до рандомизации,
учёт нескольких сессий, мощности ограничения и сетевых эффектов: один продавец
может получать контакты пользователей обеих групп. Наблюдаемая успешность симуляции
не означает, что эти риски проверены.
'''
    (reports/'decision.md').write_text(body, encoding='utf-8')
    # В дашборд попадают только агрегаты без идентификаторов пользователей.
    cols = ['variant','device','category']
    agg = mart.groupby(cols).agg(n=('user_id','size'), searched=('searched','sum'),
          viewed=('viewed','sum'), contacts=('contacted','sum'),
          funnel_contacts=('funnel_contacted','sum'), complaints=('complained','sum')).reset_index()
    records = json.dumps(agg.to_dict(orient='records'))
    template = (ROOT/'src/dashboard.html').read_text(encoding='utf-8')
    (reports/'dashboard.html').write_text(template.replace('__DATA__', records),encoding='utf-8')


def main():
    plan = json.loads((ROOT/'experiment_plan.json').read_text(encoding='utf-8'))
    data, reports = ROOT/'data', ROOT/'reports'
    data.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    a, raw = generate(plan)
    a.to_csv(data/'assignments.csv',index=False)
    raw.to_csv(data/'events_raw.csv',index=False)
    clean, quarantine, audit = clean_events(a,raw)
    quarantine.to_csv(data/'quarantine.csv',index=False)
    mart = build_mart(a,clean,data/'analytics.sqlite')
    assert len(mart) == len(a) and mart.user_id.is_unique
    assert (mart.searched >= mart.viewed).all()
    assert (mart.viewed >= mart.funnel_contacted).all()
    result = analyze(mart,plan)
    mart.to_csv(data/'user_mart.csv',index=False)
    for name, obj in [('results',result),('quality',audit)]:
        (reports/f'{name}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    write_charts(mart,result,reports)
    write_report(mart,result,audit,reports)
    print('Анализ завершён. Отчёты сохранены в папке reports.')
    print('Решение:', DECISION_LABELS[result['decision']])


if __name__ == '__main__':
    main()
