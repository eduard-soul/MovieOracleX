import asyncio
import logging
import sys
import time
import aiohttp
from lxml.html import HtmlElement, document_fromstring
import csv
import os

# Configuration constants
LIMIT_PER_HOST = 60
MAX_POPULAR_USER_CONCURRENCY = 4
BASE_URL = "http://letterboxd.com/"
ALLTIME_POPULAR_URL = BASE_URL + "members/popular/this/all-time/page/{page}/"
MONTHLY_POPULAR_URL = BASE_URL + "members/popular/this/month/{page}"
YEARLY_POPULAR_URL = BASE_URL + "members/popular/this/year/{page}/"
WEEKLY_POPULAR_URL = BASE_URL + "members/popular/this/week/page/{page}/"
PAGE = 102

# Logging setup
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()
logging.getLogger("asyncio").setLevel(logging.INFO)

async def fetch(client: aiohttp.ClientSession, url: str) -> HtmlElement:
    async with client.get(url) as resp:
        return document_fromstring(await resp.text())

async def get_users_from_page(client: aiohttp.ClientSession, movie_slug: str, page: int, sem: asyncio.Semaphore) -> list[str]:
    async with sem:
        url = BASE_URL + f"film/{movie_slug}/members/page/{page}/"  # Modified to use dynamic page
        try:
            doc = await fetch(client, url)
            log.debug(f"Fetched page {page} for movie '{movie_slug}'")
            user_links = doc.cssselect("div.person-summary a.avatar")
            usernames = [link.get("href").strip("/") for link in user_links]
            log.debug(f"Found {len(usernames)} users on page {page}")
            return usernames
        except Exception as e:
            log.error(f"Error fetching page {page} for movie '{movie_slug}': {str(e)}")
            return []

async def get_users_from_movie_members(client: aiohttp.ClientSession, movie_slug: str, sem: asyncio.Semaphore, add_user: callable, add_less_25_movie: callable, page: int) -> None:  # Added page parameter
    async with sem:
        url = BASE_URL + f"film/{movie_slug}/members/page/{page}"  # Modified to use dynamic page
        try:
            doc = await fetch(client, url)
            log.debug(f"Fetched page {page} for movie '{movie_slug}'")
        except Exception as e:
            log.error(f"Failed to fetch page {page} for movie '{movie_slug}': {str(e)}")
            return

    user_links = doc.cssselect("div.person-summary a.avatar")
    usernames = [link.get("href").strip("/") for link in user_links]
    log.debug(f"Found {len(usernames)} users on page {page}")

    if len(usernames) < 25:
        await add_less_25_movie(movie_slug)

    pages = doc.cssselect("div.pagination div.paginate-pages ul li a")
    page_numbers = [int(a.text_content()) for a in pages if a.text_content().isdigit()]
    total_pages = max(page_numbers) if page_numbers else 1
    log.debug(f"Total pages for movie '{movie_slug}': {total_pages}")

    all_usernames = usernames
    tasks = []
    for p in range(2, total_pages + 1):
        tasks.append(asyncio.create_task(get_users_from_page(client, movie_slug, p, sem)))
    results = await asyncio.gather(*tasks)
    for result in results:
        all_usernames.extend(result)

    unique_usernames = list(set(all_usernames))
    log.info(f"Found {len(unique_usernames)} unique users from movie '{movie_slug}' on page {page}")

    for user in unique_usernames:
        await add_user(user)

async def put_users(client: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore, add_user: callable) -> None:
    async with sem:
        try:
            doc = await fetch(client, url)
            els = doc.cssselect("table.person-table a.name")
            users = [el.get("href").strip("/") for el in els]
            for user in users:
                await add_user(user)
        except Exception as e:
            log.error(f"Failed to fetch popular users from {url}: {str(e)}")

async def main(current_page: int):  # Added current_page parameter
    less_25_movies = set()
    less_25_lock = asyncio.Lock()

    film_ids = []
    with open('unique_film_ids.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            film_ids.append(row['film_id'].strip())

    existing_users = set()
    csv_file = 'users.csv'
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    existing_users.add(row[0])

    with open(csv_file, 'a', newline='') as f, open('less_25_film_ids.csv', 'a', newline='') as f_less:
        csv_writer = csv.writer(f)
        csv_writer_less = csv.writer(f_less)
        csv_lock = asyncio.Lock()
        less_lock = asyncio.Lock()

        async def add_user(user: str):
            async with csv_lock:
                if user not in existing_users:
                    existing_users.add(user)
                    csv_writer.writerow([user])
                    f.flush()
                    log.debug(f"Added new user: {user}")

        async def add_less_25_movie(movie_slug: str):
            async with less_lock:
                csv_writer_less.writerow([movie_slug])
                f_less.flush()
                log.debug(f"Added movie with less than 25 users: {movie_slug}")
            async with less_25_lock:
                less_25_movies.add(movie_slug)

        http_conn = aiohttp.TCPConnector(limit_per_host=LIMIT_PER_HOST)
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(connector=http_conn, timeout=timeout) as client:
            sem = asyncio.Semaphore(MAX_POPULAR_USER_CONCURRENCY)

            movie_tasks = [
                asyncio.create_task(get_users_from_movie_members(client, film_id, sem, add_user, add_less_25_movie, current_page))  # Pass current_page
                for film_id in film_ids
            ]

            urls = [
                url.format(page=page) for page in range(1, 257)
                for url in [ALLTIME_POPULAR_URL, WEEKLY_POPULAR_URL, MONTHLY_POPULAR_URL, YEARLY_POPULAR_URL]
            ]
            popular_tasks = [
                asyncio.create_task(put_users(client, url, sem, add_user))
                for url in urls
            ]

            await asyncio.gather(*movie_tasks, *popular_tasks)

    updated_film_ids = [film_id for film_id in film_ids if film_id not in less_25_movies]
    with open('unique_film_ids.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['film_id'])
        writer.writeheader()
        for film_id in updated_film_ids:
            writer.writerow({'film_id': film_id})
    log.info(f"Updated unique_film_ids.csv, removed {len(less_25_movies)} movies")

if __name__ == "__main__":
    log.info("Starting scrape")
    current_page = PAGE
    while True:  # Infinite loop
        start_time = time.time()
        asyncio.run(main(current_page))
        log.info(f"Completed page {current_page} in {time.time() - start_time}s")
        current_page = current_page + 2  # Previous page + 2
        log.info(f"Starting next iteration with page {current_page}")
