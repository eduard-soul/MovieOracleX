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
MAX_POPULAR_USER_CONCURRENCY = 4  # Adjusted to control concurrency
BASE_URL = "http://letterboxd.com/"
ALLTIME_POPULAR_URL = BASE_URL + "members/popular/this/all-time/page/{page}/"
MONTHLY_POPULAR_URL = BASE_URL + "members/popular/this/month/{page}"
YEARLY_POPULAR_URL = BASE_URL + "members/popular/this/year/{page}/"
WEEKLY_POPULAR_URL = BASE_URL + "members/popular/this/week/page/{page}/"

# Logging setup
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()
logging.getLogger("asyncio").setLevel(logging.INFO)

# Fetch HTML content from a URL
async def fetch(client: aiohttp.ClientSession, url: str) -> HtmlElement:
    async with client.get(url) as resp:
        return document_fromstring(await resp.text())

# Fetch users from a specific page of a movie's members
async def get_users_from_page(client: aiohttp.ClientSession, movie_slug: str, page: int, sem: asyncio.Semaphore) -> list[str]:
    async with sem:
        url = BASE_URL + f"film/{movie_slug}/members/page/3/"
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

# Collect users from all pages of a movie's members
async def get_users_from_movie_members(client: aiohttp.ClientSession, movie_slug: str, sem: asyncio.Semaphore, add_user: callable) -> None:
    async with sem:
        url = BASE_URL + f"film/{movie_slug}/members/"
        try:
            doc = await fetch(client, url)
            log.debug(f"Fetched first page for movie '{movie_slug}'")
        except Exception as e:
            log.error(f"Failed to fetch first page for movie '{movie_slug}': {str(e)}")
            return

    user_links = doc.cssselect("div.person-summary a.avatar")
    usernames = [link.get("href").strip("/") for link in user_links]
    log.debug(f"Found {len(usernames)} users on page 1")

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
    log.info(f"Found {len(unique_usernames)} unique users from movie '{movie_slug}'")

    for user in unique_usernames:
        await add_user(user)

# Collect users from popular users pages
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

# Main execution
async def main():
    # Read film IDs from CSV
    film_ids = []
    with open('unique_film_ids.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            film_ids.append(row['film_id'].strip())

    # Initialize set of existing users
    existing_users = set()
    csv_file = 'users.csv'
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Ensure row is not empty
                    existing_users.add(row[0])

    # Open CSV file for appending and set up writer
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_lock = asyncio.Lock()

        # Function to add a new user to the CSV
        async def add_user(user: str):
            async with csv_lock:
                if user not in existing_users:
                    existing_users.add(user)
                    csv_writer.writerow([user])
                    f.flush()  # Ensure immediate write
                    log.debug(f"Added new user: {user}")

        # Set up HTTP client
        http_conn = aiohttp.TCPConnector(limit_per_host=LIMIT_PER_HOST)
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(connector=http_conn, timeout=timeout) as client:
            sem = asyncio.Semaphore(MAX_POPULAR_USER_CONCURRENCY)

            # Tasks for movie members
            movie_tasks = [
                asyncio.create_task(get_users_from_movie_members(client, film_id, sem, add_user))
                for film_id in film_ids
            ]

            # Tasks for popular users
            urls = [
                url.format(page=page) for page in range(1, 257)
                for url in [ALLTIME_POPULAR_URL, WEEKLY_POPULAR_URL, MONTHLY_POPULAR_URL, YEARLY_POPULAR_URL]
            ]
            popular_tasks = [
                asyncio.create_task(put_users(client, url, sem, add_user))
                for url in urls
            ]

            # Execute all tasks concurrently
            await asyncio.gather(*movie_tasks, *popular_tasks)

if __name__ == "__main__":
    log.info("Starting scrape")
    start_time = time.time()
    asyncio.run(main())
    log.info(f"Completed in {time.time() - start_time}s")
