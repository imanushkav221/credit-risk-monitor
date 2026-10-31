"""Visit the hosted app and wake it if Streamlit put it to sleep (used by .github/workflows/keep-awake.yml)."""
from playwright.sync_api import sync_playwright

URL = "https://credit-risk-monitor-el.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, timeout=120_000)
    page.wait_for_timeout(15_000)
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count():
        print("App was asleep, waking it")
        wake.click()
        page.wait_for_timeout(90_000)
    else:
        print("App is awake")
    browser.close()
