import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.error import URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ddgs import DDGS
from ddgs.exceptions import DDGSException

FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook"
FORECAST_SOURCE = "https://developers.data.gov.sg/datasets/d_f131f6e343bf8168e4057a04c4326a0a/view"
CNA_SINGAPORE_FEED = "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=10416"


def singapore_headlines() -> list[dict[str, str]]:
    request = Request(CNA_SINGAPORE_FEED, headers={"User-Agent": "LocalVoiceAssistant/1.0"})
    try:
        with urlopen(request, timeout=8) as response:
            feed = ElementTree.fromstring(response.read(1_000_000))
    except (URLError, TimeoutError, ElementTree.ParseError) as error:
        raise RuntimeError("I couldn’t reach the CNA Singapore headlines feed. Try again shortly.") from error
    sources = []
    for item in feed.findall("./channel/item"):
        url = item.findtext("link", "").strip()
        title = item.findtext("title", "").strip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate", ""))
        except (TypeError, ValueError):
            continue
        if title and urlsplit(url).hostname == "www.channelnewsasia.com" and recent(published):
            sources.append({"url": url, "title": title})
        if len(sources) == 4:
            return sources
    if not sources:
        raise RuntimeError("I couldn’t find recent articles in the CNA Singapore feed.")
    return sources


def recent(published: datetime) -> bool:
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return timedelta(0) <= datetime.now(timezone.utc) - published <= timedelta(days=7)


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


def search_web(query: str, news: bool = False) -> list[dict[str, str]]:
    search = DDGS(timeout=5)
    results = []
    fetch = search.news if news else search.text
    engines = ("bing", "duckduckgo", "yahoo") if news else ("bing", "duckduckgo")
    for engine in engines:
        try:
            filters = {"timelimit": "w"} if news else {}
            results = fetch(query, max_results=6 if news else 4, backend=engine, **filters)
        except DDGSException:
            continue
        if results:
            break

    sources = []
    seen_urls = set()
    for result in results:
        url = result.get("url") or result.get("href", "")
        parts = urlsplit(url)
        if url in seen_urls or (news and (not parts.path.strip("/") or not result.get("date"))):
            continue
        if news:
            try:
                if not recent(datetime.fromisoformat(result["date"])):
                    continue
            except (TypeError, ValueError):
                continue
        if parts.scheme in ("https", "http") and parts.hostname and not parts.username:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "title": result.get("title", "")[:200],
                "snippet": result.get("body", "")[:800],
            })
            if news:
                sources[-1]["published"] = result["date"]
            if len(sources) == 4:
                break
    if not sources:
        if news:
            raise RuntimeError("I couldn’t find recent news articles for that query. Try a specific topic or location.")
        raise RuntimeError("Web search found no sources. Try a shorter query or send it with search off.")
    return sources


def without_source_section(reply: str) -> str:
    return re.split(
        r"(?im)^\s*(?:#{1,6}\s+)?\*{0,2}(?:sources?|references)\*{0,2}\s*:?\*{0,2}[ \t]*$",
        reply,
        maxsplit=1,
    )[0].rstrip()


def source_links(sources: list[dict[str, str]]) -> str:
    links = []
    for index, source in enumerate(sources, 1):
        url = quote(source["url"], safe=":/?#[]@!$&'+,;=%")
        title = source.get("title") or urlsplit(url).hostname
        title = re.sub(r"([\\\[\]<>*_])", r"\\\1", " ".join(title.split()))
        links.append(f"- [{index}. {title}](<{url}>)")
    return "\n\n**Sources**\n\n" + "\n".join(links)
