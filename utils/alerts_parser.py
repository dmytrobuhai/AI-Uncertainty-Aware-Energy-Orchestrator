from playwright.sync_api import sync_playwright
from datetime import datetime
import time
import os
from pathlib import Path

START_DATE = datetime(2025, 12, 19)
END_DATE   = datetime(2022, 5, 1)

BASE = Path(__file__).parent.parent
DOWNLOAD_DIR = Path.joinpath(BASE, Path("data"), Path("alerts"))
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def month_year_from_title(title: str) -> datetime:
    months = {
        "січень": 1, "лютий": 2, "березень": 3, "квітень": 4,
        "травень": 5, "червень": 6, "липень": 7, "серпень": 8,
        "вересень": 9, "жовтень": 10, "листопад": 11, "грудень": 12
    }
    m, y = title.split()
    return datetime(int(y), months[m], 1)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    page.goto("https://alerts.in.ua/?showThreats&showWarnings", timeout=60000)

    # Відкрити історію
    page.locator("i.fa-clock-rotate-left").click()
    page.wait_for_timeout(1500)

    # === 1. Докрутити календар ДО START_DATE ===
    while True:
        title = page.locator(".vc-title").inner_text()
        current_month = month_year_from_title(title)

        if current_month <= START_DATE:
            break

        page.locator("path[d^='M11.196 10c']").click()
        page.wait_for_timeout(1200)

    # === 2. Основний цикл парсингу ===
    while True:
        title = page.locator(".vc-title").inner_text()
        current_month = month_year_from_title(title)

        print(f"=== {title} ===")

        if current_month < END_DATE:
            print("Досягнуто END_DATE. Стоп.")
            break

        days = page.locator(
            ".vc-day.in-month span.vc-day-content:not(.is-disabled)"
        )

        count = days.count()

        for i in range(count):
            day = days.nth(i)
            day_div = day.locator("..")

            class_attr = day_div.get_attribute("class")

            date_str = next(
                part.replace("id-", "")
                for part in class_attr.split()
                if part.startswith("id-")
            )

            day_date = datetime.strptime(date_str, "%Y-%m-%d")

            if not (END_DATE <= day_date <= START_DATE):
                continue

            print("Клік:", day_date.date())

            day.click()
            page.wait_for_timeout(400)

            with page.expect_download() as download_info:
                page.locator(
                    "a.no-underline:has(i.fa-download)"
                ).click()

            download = download_info.value
            download.save_as(os.path.join(DOWNLOAD_DIR, download.suggested_filename))

            time.sleep(0.4)


        # Попередній місяць
        page.locator("path[d^='M11.196 10c']").click()
        page.wait_for_timeout(1200)

    browser.close()
