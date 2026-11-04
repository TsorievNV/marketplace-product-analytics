import json
import numpy as np
import pandas as pd
import pytest
from src.pipeline import (ROOT, generate, clean_events, build_mart, compare_rates,
                          required_n, analyze)


def test_simulation_faults_and_denominator():
    plan=json.loads((ROOT/'experiment_plan.json').read_text(encoding='utf-8'))
    plan['users']=1200
    a, raw=generate(plan)
    clean, quarantine, audit=clean_events(a,raw)
    assert len(quarantine)==45
    assert audit['unknown_user_rows']==15
    assert audit['invalid_timestamp_rows']==15
    assert audit['missing_listing_rows']==15
    assert clean.event_id.is_unique
    mart=build_mart(a,clean)
    assert len(mart)==len(a) and mart.user_id.is_unique
    assert (mart.searched>=mart.viewed).all()
    assert (mart.viewed>=mart.funnel_contacted).all()
    assert (mart.contacted==mart.funnel_contacted).all()


def fixture_mart():
    a=pd.DataFrame({'user_id':[1,2,3,4], 'variant':['A','A','B','B'],
                    'assigned_at':['2026-10-01']*4,'window_end':['2026-10-08']*4})
    e=pd.DataFrame([
       [2,'s2','4','contact','2026-10-02'], # Контакт без предыдущих шагов входит в ITT, но не в воронку.
       [3,'s3','5','search','2026-10-02'],
       [3,'s3','5','listing_view','2026-10-03'],
       [3,'s3','6','contact','2026-10-04'], # Контакт по другому объявлению.
       [4,'s4','7','search','2026-10-02'],
       [4,'s4','7','listing_view','2026-10-03'],
       [4,'s4','7','contact','2026-10-09'], # Контакт после семидневного окна.
    ],columns=['user_id','session_id','listing_id','event_name','event_time'])
    return build_mart(a,e).set_index('user_id')


def test_no_events_does_not_drop_user():
    m=fixture_mart()
    assert len(m)==4 and m.loc[1,'contacted']==0


def test_orphan_and_wrong_listing_not_in_funnel():
    m=fixture_mart()
    assert m.loc[2,'contacted']==1 and m.loc[2,'funnel_contacted']==0
    assert m.loc[3,'contacted']==1 and m.loc[3,'funnel_contacted']==0


def test_late_conversion_excluded():
    m=fixture_mart()
    assert m.loc[4,'contacted']==0


def test_conflicting_duplicate_fails():
    p=json.loads((ROOT/'experiment_plan.json').read_text(encoding='utf-8'));p['users']=100
    a,r=generate(p)
    r.loc[r.index[-1],'event_id']=r.iloc[0].event_id
    with pytest.raises(ValueError, match='Разные данные'):
        clean_events(a,r)


def test_rates_and_power():
    result=compare_rates(200,1000,250,1000)
    assert result['difference']==pytest.approx(.05)
    assert result['ci_low']<.05<result['ci_high']
    assert result['pvalue']<.01
    assert required_n(delta=.009)>3.5*required_n(delta=.018)


def test_guardrail_uncertainty_is_not_safety():
    r=compare_rates(8,1000,9,1000)
    assert r['pvalue']>.05
    assert r['upper_one_sided_95']>.003


def test_aa_false_positive_rate():
    rng=np.random.default_rng(42)
    rejected=0
    for _ in range(1000):
        x,y=rng.binomial(10000,.22,size=2)
        rejected+=compare_rates(x,10000,y,10000)['pvalue']<.05
    # Допуск для проверки методом Монте-Карло; он не доказывает корректность всех допущений.
    assert .025<rejected/1000<.075
