import requests
import os
import re
from urllib.parse import urlparse, unquote
from typing import Optional, List


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
    """
    Extract the filename from a URL.
    Example:
        https://.../9933-gol-jaz-tc-fm.mp4 -> 9933-gol-jaz-tc-fm.mp4
    """
    path = urlparse(url).path
    name = os.path.basename(unquote(path))
    return name if name else "downloaded_file.mp4"


def download_with_proxy(
    url: str,
    proxy_list: List[str],
    output_filename: str,
    connect_timeout: int = 10,
    read_timeout: int = 60
) -> Optional[str]:
    """Try each proxy until one successfully downloads the file."""
    if not proxy_list:
        print("❌ No proxies to test.")
        return None

    print(f"\n🔍 Testing {len(proxy_list)} proxies...")

    for i, raw_proxy in enumerate(proxy_list, 1):
        proxy = raw_proxy if raw_proxy.startswith("socks5://") else f"socks5://{raw_proxy}"
        proxies = {"http": proxy, "https": proxy}

        print(f"\n[{i}/{len(proxy_list)}] Testing: {raw_proxy}")

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
                if total_size > 0:
                    print(f"   📦 Size: {total_size / (1024 * 1024):.2f} MB")

                downloaded = 0
                chunk_size = 65536

                with open(output_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                if downloaded == 0:
                    print("   ❌ Empty file received.")
                    os.remove(output_filename)
                    continue

                if total_size > 0 and downloaded < total_size * 0.99:
                    print(f"   ⚠️ Incomplete: {downloaded}/{total_size} bytes")
                    os.remove(output_filename)
                    continue

                print(f"   ✅ Success! ({downloaded / (1024 * 1024):.2f} MB)")
                return raw_proxy

        except requests.exceptions.ProxyError as e:
            print(f"   ❌ Proxy error: {str(e)[:100]}")
        except requests.exceptions.ConnectTimeout:
            print("   ❌ Connect timeout")
        except requests.exceptions.ReadTimeout:
            print("   ❌ Read timeout")
        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP error: {e}")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {str(e)[:100]}")

        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
            except Exception:
                pass

    print("\n❌ No working proxy found.")
    return None


def auto_download(url: str, output_filename: Optional[str] = None) -> Optional[str]:
    """
    Fetch proxies, test them, and download the file.
    If output_filename is None, the filename is auto-derived from the URL.
    """
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