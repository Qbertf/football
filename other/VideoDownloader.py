import os
import json
import subprocess
import shutil
from pathlib import Path


class VideoDownloader:
    def __init__(self):
        """
        Initialize the Video Downloader
        """
        self.results = {}
        self.repo_dir = 'shopare_repo'  # Directory for cloning

    def clone_repo_and_get_backup(self):
        """
        Clone the repository and get the sessions_backup.json file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove existing directory if it exists
            if os.path.exists(self.repo_dir):
                print(f"🗑️ Removing existing directory: {self.repo_dir}")
                shutil.rmtree(self.repo_dir)

            # Clone the repository with readonly branch
            print("📥 Cloning repository from GitHub...")
            repo_url = "https://github.com/metrirun/shopare.git"
            clone_cmd = ['git', 'clone', '--branch', 'readonly', '--single-branch', repo_url, self.repo_dir]

            result = subprocess.run(clone_cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"❌ Git clone failed: {result.stderr}")
                return False

            print("✅ Repository cloned successfully")

            # Check if the backup file exists
            backup_path = Path(self.repo_dir) / 'sessions_backup.json'
            if backup_path.exists():
                print(f"✅ Found sessions_backup.json in repository")
                # Copy the file to current directory
                shutil.copy(backup_path, 'sessions_backup.json')
                print("✅ sessions_backup.json copied to current directory")
                return True
            else:
                print(f"❌ sessions_backup.json not found in repository")
                return False

        except Exception as e:
            print(f"❌ Error cloning repository: {str(e)}")
            return False

    def download_from_google(self, link):
        """
        Download a file from Google Drive using gdown

        Args:
            link: Google Drive file ID or URL

        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Check if gdown is installed
            subprocess.run(['gdown', '--version'], capture_output=True, check=True)

            # Download the file
            result = subprocess.run(['gdown', link, '-O', link],
                                    capture_output=True, check=False)

            if result.returncode == 0:
                print(f"✅ Google Drive download successful for {link}")
                return True
            else:
                error_msg = result.stderr.decode()
                print(f"❌ Google Drive download failed for {link}: {error_msg}")
                return False

        except subprocess.CalledProcessError:
            print("❌ gdown not installed. Please install: pip install gdown")
            return False
        except Exception as e:
            print(f"❌ Error downloading from Google Drive: {str(e)}")
            return False

    def unscramble_json(self, scrambled_bytes, password):
        """
        Unscramble JSON data using XOR with password

        Args:
            scrambled_bytes: Bytes to unscramble
            password: Password string for XOR

        Returns:
            dict: Unscrambled JSON data
        """
        try:
            password_bytes = str(password).encode('utf-8')
            unscrambled = bytearray()

            for i in range(len(scrambled_bytes)):
                unscrambled.append(scrambled_bytes[i] ^ password_bytes[i % len(password_bytes)])

            return json.loads(unscrambled.decode('utf-8'))
        except Exception as e:
            print(f"❌ Error unscrambling JSON: {str(e)}")
            return {}

    def process_sessions_backup(self, backup_file='sessions_backup.json'):
        """
        Process the sessions backup file to extract drive links.
        Returns:
            dict: Dictionary mapping MFO to a tuple (link, is_public, generated_id, plink)
        """
        drive_links = {}

        if not os.path.exists(backup_file):
            print(f"❌ File '{backup_file}' not found!")
            return drive_links

        try:
            with Path(backup_file).open("rb") as f:
                loaded_data = f.read()

            original_data = self.unscramble_json(loaded_data, "armeji")
            print("✅ Data successfully unscrambled!")

            for item in original_data:
                try:
                    all_info_str = item.get('all_info', '{}').replace('null', 'None')
                    all_info = eval(all_info_str)

                    match_details = all_info.get('match_details', {})
                    mfo = match_details.get('Match Folder')
                    link = match_details.get('Google Drive Link')
                    plink = match_details.get('Public Video Link')

                    # اگر mfo نبود، از plink یک نام بساز
                    if not mfo and plink:
                        mfo = os.path.basename(plink).split('.')[0]

                    if not mfo:
                        continue

                    # حالت اول: لینک گوگل درایو معتبر است
                    if link and 'drive' in link:
                        try:
                            link_id = link.split('/')[-2]
                            drive_links[mfo] = (link_id, False, None, None)
                            continue
                        except Exception:
                            link = None  # لینک خراب

                    # حالت دوم: لینک گوگل نبود یا خراب بود -> از plink استفاده کن
                    if plink:
                        import hashlib
                        generated_id = hashlib.md5(plink.encode()).hexdigest()[:33]
                        local_filename = f"{mfo}.mp4"
                        drive_links[mfo] = (local_filename, True, generated_id, plink)

                except Exception as e:
                    print(f"⚠️ Error processing item: {e}")
                    continue

            print(f"✅ Found {len(drive_links)} links in backup")
            return drive_links

        except Exception as e:
            print(f"❌ Error processing backup file: {str(e)}")
            return drive_links

    def process_all_links(self):
        """
        Process all links from the sessions backup file.
        Returns:
            dict: Dictionary with results for each link
        """
        print("📥 Getting sessions_backup.json from GitHub repository...")
        if not self.clone_repo_and_get_backup():
            print("❌ Failed to get backup file from repository")
            return {}

        drive_links = self.process_sessions_backup()

        if not drive_links:
            print("❌ No links found to process")
            return {}

        print("\n" + "=" * 60)
        print("🚀 Starting download process...")
        print("=" * 60 + "\n")

        for key, (link, is_public, generated_id, plink) in drive_links.items():
            # تعیین نام ویدئو
            if is_public:
                video_id = generated_id.replace('_', '-')
                print(f"📁 Processing: {key} -> Public link (ID: {video_id})")
            else:
                video_id = link.replace('_', '-')
                print(f"📁 Processing: {key} -> {link} (ID: {video_id})")

            flag_google = False

            # فقط دانلود از گوگل درایو
            if not is_public:
                # حالت گوگل درایو
                if not os.path.exists(link):
                    print(f"📥 File {link} not found locally. Downloading from Google Drive...")
                    flag_google = self.download_from_google(link)
                else:
                    print(f"📄 File {link} found locally.")
                    flag_google = True
            else:
                # حالت لینک عمومی - فعلاً پشتیبانی نمی‌شود
                print(f"⚠️ Public link download not supported in this version")
                flag_google = False

            # ذخیره نتایج - ویدئو به صورت ID ذخیره می‌شود
            self.results[video_id] = {
                'mfo': key,
                'flag_google': flag_google,
                'original_name': link if not is_public else 'public_link',
                'video_id': video_id
            }

            # نمایش وضعیت
            google_status = "✅" if flag_google else "❌"

            print(f"\n📊 Results for {video_id} (original: {link if not is_public else 'public_link'}):")
            print(f"   ☁️ Google:  {google_status} {'Downloaded' if flag_google else 'Download Failed'}")
            print(f"   🆔 Video ID: {video_id}")
            print("-" * 40 + "\n")

        return self.results

    def get_results(self):
        """
        Get the results of the download process

        Returns:
            dict: Results dictionary
        """
        return self.results


def main():
    """
    Main function to run the Video Downloader

    Returns:
        tuple: (flag_google, dataset_mapping)
            - flag_google: bool
            - dataset_mapping: dict {MatchName: video_id}
    """
    print("🚀 Starting Video Downloader (Download Only Mode)...")

    # ✅ دیکشنری خروجی مورد نظر شما
    dataset_mapping = {}

    try:
        # Create downloader instance
        downloader = VideoDownloader()

        # Process all links
        results = downloader.process_all_links()

        # Get final summary
        if results:
            print("\n" + "=" * 60)
            print("📊 FINAL SUMMARY")
            print("=" * 60)

            for video_id, data in results.items():
                google_status = "✅" if data['flag_google'] else "❌"
                print(f"📁 {data['mfo']}: Google {google_status} | Video ID: {video_id} (original: {data.get('original_name', 'N/A')})")

                # ✅ پر کردن دیکشنری: Matchname -> video_id
                match_name = data['mfo']  # نام مسابقه
                dataset_mapping[match_name] = data['video_id']  # ذخیره video_id

            print("=" * 60)

            # Return the last processed link's flag + mapping
            last_result = list(results.values())[-1] if results else {'flag_google': False}
            return (last_result['flag_google'], dataset_mapping)
        else:
            print("❌ No results to display")
            return (False, {})

    except Exception as e:
        print(f"❌ Error in main process: {str(e)}")
        return (False, {})


if __name__ == "__main__":
    flag_google, dataset_mapping = main()
    print(f"\n🏁 FINAL RESULTS:")
    print(f"☁️ Google Flag: {'✅ SUCCESS' if flag_google else '❌ FAILED'}")
    print(f"📊 Dataset Mapping: {dataset_mapping}")
