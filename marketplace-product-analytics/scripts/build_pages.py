"""Подготовка HTML-дашборда для GitHub Pages."""
import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DECISION_URL = (
    'https://github.com/TsorievNV/marketplace-product-analytics/'
    'blob/main/marketplace-product-analytics/reports/decision.md'
)


def build(output):
    html = (PROJECT / 'reports/dashboard.html').read_text(encoding='utf-8')
    if '__DATA__' in html:
        raise ValueError('Сначала выполните анализ: python -m src.pipeline')
    if 'href="decision.md"' not in html:
        raise ValueError('Не найдена ссылка на аналитическую записку')
    html = html.replace('href="decision.md"', f'href="{DECISION_URL}"')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'index.html').write_text(html, encoding='utf-8')
    print('Дашборд подготовлен:', output / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('_site'))
    build(parser.parse_args().output)
