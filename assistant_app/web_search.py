import json
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from ddgs import DDGS
from ddgs.exceptions import DDGSException

FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook"
FORECAST_SOURCE = "https://developers.data.gov.sg/datasets/d_f131f6e343bf8168e4057a04c4326a0a/view"


def singapore_tomorrow() -> str:
    singapore = timezone(timedelta(hours=8))
    tomorrow = datetime.now(singapore).date() + timedelta(days=1)
    try:
        request = Request(FORECAST_URL, headers={
            "Accept": "application/json", "User-Agent": "LocalVoiceAssistant/1.0",
        })
        with urlopen(request, timeout=8) as response:
            records = json.load(response)["data"]["records"]
        forecast = next(
            item for record in records for item in record["forecasts"]
            if item["timestamp"][:10] == tomorrow.isoformat()
        )
        summary = forecast["forecast"]["summary"]
        low = forecast["temperature"]["low"]
        high = forecast["temperature"]["high"]
    except (URLError, TimeoutError, ValueError, KeyError, TypeError, StopIteration) as error:
        raise RuntimeError("Could not get tomorrow's Singapore forecast from NEA.") from error
    return (
        f"Tomorrow in Singapore, NEA forecasts {summary.lower()}, "
        f"with temperatures from {low} to {high} degrees Celsius."
    )


def search_web(query: str) -> list[dict[str, str]]:
    search = DDGS(timeout=5)
    results = []
    for engine in ("bing", "duckduckgo"):
        try:
            results = search.text(query, max_results=4, backend=engine)
        except DDGSException:
            continue
        if results:
            break

    sources = []
    for result in results:
        url = result.get("href", "")
        parts = urlsplit(url)
        if parts.scheme in ("https", "http") and parts.hostname and not parts.username:
            sources.append({
                "url": url,
                "title": result.get("title", "")[:200],
                "snippet": result.get("body", "")[:800],
            })
    if not sources:
        raise RuntimeError("Web search found no sources. Try a shorter query or send it with search off.")
    return sources


def source_links(sources: list[dict[str, str]]) -> str:
    links = []
    for index, source in enumerate(sources, 1):
        url = quote(source["url"], safe=":/?#[]@!$&'+,;=%")
        links.append(f"- [{index}. {urlsplit(url).hostname}](<{url}>)")
    return "\n\n**Sources**\n\n" + "\n".join(links)
