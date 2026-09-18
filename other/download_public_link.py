import requests
import os
import re
import time
from urllib.parse import urlparse, unquote
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_proxy_list(api_url: str = "https://hproxy.com/api/proxy-list?format=txt&protocol=socks5",
                     limit: int = 200) -> List[str]:
    """Fetch a list of SOCKS5 proxies from the API."""
    print("📡 Fetching proxy list...")
    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()
    except Exception as e:
        print(f"❌ Failed to fetch proxy list: {e}")
        return []

    proxies = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d{1,3}(\.\d{1,3}){3}:\d+$', line):
            proxies.append(line)
        if len(proxies) >= limit:
            break

    print(f"✅ Found {len(proxies)} proxies.")
    return proxies


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(unquote(path))
    return name if name else "downloaded_file.mp4"


def _test_proxy_speed(
    url: str,
    raw_proxy: str,
    connect_timeout: int = 8,
    read_timeout: int = 15,
    test_bytes: int = 256 * 1024,   # 256 KB نمونه
) -> Optional[Tuple[str, float, int]]:
    """
    Test a single proxy by downloading a small chunk and measuring speed.
    Returns (proxy, speed_bytes_per_sec, total_size) or None.
    """
    proxy = raw_proxy if raw_proxy.startswith("socks5://") else f"socks5://{raw_proxy}"
    proxies = {"http": proxy, "https": proxy}

    try:
        with requests.get(
            url,
            proxies=proxies,
            stream=True,
            timeout=(connect_timeout, read_timeout),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
        ) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            downloaded = 0
            start = time.time()
            for chunk in r.iter_content(chunk_size=32768):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded >= test_bytes:
                    break

            elapsed = time.time() - start
            if elapsed <= 0 or downloaded == 0:
                return None
            speed = downloaded / elapsed
            return (raw_proxy, speed, total_size)

    except Exception:
        return None


def _download_with_single_proxy(
    url: str,
    raw_proxy: str,
    output_filename: str,
    connect_timeout: int = 10,
    read_timeout: int = 60,
) -> Optional[int]:
    """Download file using a specific proxy. Returns bytes downloaded or None."""
    proxy = raw_proxy if raw_proxy.startswith("socks5://") else f"socks5://{raw_proxy}"
    proxies = {"http": proxy, "https": proxy}

    try:
        with requests.get(
            url,
            proxies=proxies,
            stream=True,
            timeout=(connect_timeout, read_timeout),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
        ) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 65536

            with open(output_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if downloaded == 0:
                os.remove(output_filename)
                return None
            if total_size > 0 and downloaded < total_size * 0.99:
                os.remove(output_filename)
                return None
            return downloaded

    except Exception:
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
            except Exception:
                pass
        return None


def download_with_proxy(
    url: str,
    proxy_list: List[str],
    output_filename: str,
    batch_size: int = 10,
    min_speed_bps: float = 100 * 1024,   # حداقل 100 KB/s برای قبولی
    max_candidates: int = 3,             # چند تا از سریع‌ترین‌ها برای دانلود نهایی
) -> Optional[str]:
    """
    Test proxies in parallel batches, pick the fastest one, then download.
    """
    if not proxy_list:
        print("❌ No proxies to test.")
        return None

    print(f"\n🔍 Testing {len(proxy_list)} proxies in parallel batches of {batch_size}...")

    fast_candidates: List[Tuple[str, float]] = []  # (proxy, speed)
    tested = 0

    for start in range(0, len(proxy_list), batch_size):
        batch = proxy_list[start:start + batch_size]
        print(f"\n📦 Batch {start // batch_size + 1} "
              f"({start + 1}-{start + len(batch)} of {len(proxy_list)})")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(_test_proxy_speed, url, p): p
                for p in batch
            }
            for fut in as_completed(futures):
                tested += 1
                result = fut.result()
                if result is None:
                    continue
                proxy, speed, total_size = result
                speed_kbps = speed / 1024
                print(f"   ✅ {proxy}  →  {speed_kbps:,.1f} KB/s")

                if speed >= min_speed_bps:
                    fast_candidates.append((proxy, speed))

        # اگه به تعداد کافی کاندید سریع داریم، از تست بقیه صرف‌نظر کن
        if len(fast_candidates) >= max_candidates:
            print(f"\n⚡ Found {len(fast_candidates)} fast proxies, stopping tests early.")
            break

    if not fast_candidates:
        print("\n❌ No proxy met the minimum speed requirement.")
        return None

    # مرتب‌سازی بر اساس سرعت (نزولی)
    fast_candidates.sort(key=lambda x: x[1], reverse=True)
    print("\n🏆 Top proxies by speed:")
    for p, s in fast_candidates[:max_candidates]:
        print(f"   {p}  →  {s / 1024:,.1f} KB/s")

    # تلاش برای دانلود با سریع‌ترین‌ها به ترتیب
    for proxy, speed in fast_candidates[:max_candidates]:
        print(f"\n⬇️  Downloading with {proxy} ({speed / 1024:,.1f} KB/s)...")
        downloaded = _download_with_single_proxy(url, proxy, output_filename)
        if downloaded:
            print(f"   ✅ Success! ({downloaded / (1024 * 1024):.2f} MB)")
            return proxy
        else:
            print(f"   ❌ Download failed with {proxy}, trying next...")

    print("\n❌ All fast proxies failed during full download.")
    return None


def auto_download(url: str, output_filename: Optional[str] = None) -> Optional[str]:
    if output_filename is None:
        output_filename = filename_from_url(url)
        print(f"📝 Auto-detected filename: {output_filename}")

    proxy_list = fetch_proxy_list(limit=200)
    if not proxy_list:
        return None

    working_proxy = download_with_proxy(
        url=url,
        proxy_list=proxy_list,
        output_filename=output_filename,
    )

    if working_proxy:
        print(f"\n🎯 Working proxy: {working_proxy}")
        print(f"📁 File saved as: {output_filename}")
    return working_proxy
