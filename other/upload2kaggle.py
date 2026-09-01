import os
import json
import subprocess
import threading
from pathlib import Path
import time
import sys
import shutil

class KaggleUploader:
    def __init__(self, acc_kaggle, kaggle_api_key):
        """
        Initialize the Kaggle uploader with account information
        
        Args:
            acc_kaggle: Kaggle username
            kaggle_api_key: Dictionary with 'username' and 'key'
        """
        self.acc_kaggle = acc_kaggle
        self.kaggle_api_key = kaggle_api_key
        
        # Set environment variables for Kaggle API
        os.environ['KAGGLE_USERNAME'] = kaggle_api_key['username']
        os.environ['KAGGLE_KEY'] = kaggle_api_key['key']
        
        # Store results
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
    
    def dataset_exists(self, username, dataset_slug, timeout=5):
        """
        Check if a dataset exists on Kaggle
        
        Args:
            username: Kaggle username
            dataset_slug: Dataset name
            timeout: Timeout in seconds
            
        Returns:
            bool: True if dataset exists, False otherwise
        """
        handle = f"{username}/{dataset_slug}"
        result = [False]
        
        def check():
            try:
                import kagglehub
                kagglehub.dataset_download(handle)
                result[0] = True
            except Exception as e:
                # Dataset doesn't exist or other error
                pass
        
        thread = threading.Thread(target=check)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        # If thread is still alive, download has started (dataset exists)
        if thread.is_alive():
            return True
        
        return result[0]
    
    def upload_to_kaggle(self, link):
        """
        Upload a file to Kaggle as a dataset
        
        Args:
            link: Dataset name/folder name
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            # Change to kaggle directory
            kaggle_dir = '/content/kaggle'
            if not os.path.exists(kaggle_dir):
                os.makedirs(kaggle_dir)
            
            os.chdir(kaggle_dir)
            
            # Move file to kaggle directory
            if os.path.exists(f'../{link}'):
                subprocess.run(['mv', f'../{link}', kaggle_dir], check=True)
            elif not os.path.exists(link):
                print(f"⚠️ File {link} not found for upload")
                return False
            
            # Initialize dataset metadata
            subprocess.run(['kaggle', 'datasets', 'init', '-p', kaggle_dir], 
                          capture_output=True, check=False)
            
            # Update metadata
            metadata_path = Path(kaggle_dir) / 'dataset-metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                
                meta['title'] = link
                meta['id'] = f"{self.acc_kaggle}/{link}"
                
                with open(metadata_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                
                # Create dataset
                result = subprocess.run(['kaggle', 'datasets', 'create', '-p', kaggle_dir, '-u'],
                                      capture_output=True, check=False)
                
                if result.returncode == 0:
                    print(f"✅ Upload successful for {link}")
                    return True
                else:
                    print(f"❌ Upload failed for {link}: {result.stderr.decode()}")
                    return False
            else:
                print(f"❌ Metadata file not found for {link}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading {link}: {str(e)}")
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
        Process the sessions backup file to extract drive links
        
        Args:
            backup_file: Path to the backup file
            
        Returns:
            dict: Dictionary mapping MFO to drive link
        """
        drive_links = {}
        
        # Check if file exists
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
                    # Parse all_info
                    all_info_str = item.get('all_info', '{}').replace('null', 'None')
                    all_info = eval(all_info_str)
                    
                    # Extract MFO and Google Drive Link
                    match_details = all_info.get('match_details', {})
                    mfo = match_details.get('Match Folder')
                    link = match_details.get('Google Drive Link')
                    
                    if mfo and link and 'drive' in link:
                        # Extract file ID from Google Drive link
                        link_id = link.split('/')[-2]
                        drive_links[mfo] = link_id
                        
                except Exception as e:
                    # Skip individual items that can't be processed
                    continue
            
            print(f"✅ Found {len(drive_links)} drive links in backup")
            return drive_links
            
        except Exception as e:
            print(f"❌ Error processing backup file: {str(e)}")
            return drive_links
    
    def process_all_links(self):
        """
        Process all links from the sessions backup file
        
        Returns:
            dict: Dictionary with results for each link
        """
        # Clone repository and get backup file
        print("📥 Getting sessions_backup.json from GitHub repository...")
        if not self.clone_repo_and_get_backup():
            print("❌ Failed to get backup file from repository")
            return {}
        
        # Process backup file
        drive_links = self.process_sessions_backup()
        
        if not drive_links:
            print("❌ No drive links found to process")
            return {}
        
        # Process each link
        print("\n" + "="*60)
        print("🚀 Starting upload process...")
        print("="*60 + "\n")
        
        for key, link in drive_links.items():
            print(f"📁 Processing: {key} -> {link}")
            
            flag_kaggle = False
            flag_google = False
            
            # Check if dataset exists on Kaggle
            print(f"🔍 Checking if {link} exists on Kaggle...")
            flag_kaggle = self.dataset_exists(self.acc_kaggle, link, timeout=5)
            
            if flag_kaggle:
                print(f"✅ Dataset {link} already exists on Kaggle")
            else:
                print(f"❌ Dataset {link} not found on Kaggle")
                
                # Check if file exists locally
                if not os.path.exists(link):
                    print(f"📥 File {link} not found locally. Downloading from Google Drive...")
                    flag_google = self.download_from_google(link)
                    
                    if flag_google:
                        print(f"📤 Uploading {link} to Kaggle...")
                        flag_kaggle = self.upload_to_kaggle(link)
                    else:
                        print(f"❌ Cannot proceed with upload - Google Drive download failed")
                else:
                    print(f"📄 File {link} found locally. Uploading to Kaggle...")
                    flag_kaggle = self.upload_to_kaggle(link)
            
            # Store results
            self.results[link] = {
                'mfo': key,
                'flag_kaggle': flag_kaggle,
                'flag_google': flag_google
            }
            
            # Display results with colored indicators
            kaggle_status = "✅" if flag_kaggle else "❌"
            google_status = "✅" if flag_google else "❌" if not flag_kaggle else "⏭️"
            
            print(f"\n📊 Results for {link}:")
            print(f"   📤 Kaggle:  {kaggle_status} {'Exists/Uploaded' if flag_kaggle else 'Not Found/Upload Failed'}")
            print(f"   ☁️ Google:  {google_status} {'Downloaded' if flag_google else 'Not Downloaded' if not flag_kaggle else 'Already on Kaggle'}")
            print("-"*40 + "\n")
        
        return self.results
    
    def get_results(self):
        """
        Get the results of the upload process
        
        Returns:
            dict: Results dictionary
        """
        return self.results

def main(acc_kaggle, kaggle_api_key):
    """
    Main function to run the Kaggle uploader
    
    Args:
        acc_kaggle: Kaggle username
        kaggle_api_key: Dictionary with 'username' and 'key'
        
    Returns:
        tuple: (flag_kaggle, flag_google) for the last processed link
    """
    print("🚀 Starting Kaggle Uploader...")
    print(f"📊 Account: {kaggle_api_key['username']}")
    
    # Validate API key
    if not kaggle_api_key.get('username') or not kaggle_api_key.get('key'):
        print("❌ Invalid Kaggle API credentials")
        return (False, False)
    
    try:
        # Create uploader instance
        uploader = KaggleUploader(acc_kaggle, kaggle_api_key)
        
        # Process all links
        results = uploader.process_all_links()
        
        # Get final summary
        if results:
            print("\n" + "="*60)
            print("📊 FINAL SUMMARY")
            print("="*60)
            
            for link, data in results.items():
                kaggle_status = "✅" if data['flag_kaggle'] else "❌"
                google_status = "✅" if data['flag_google'] else "❌" if not data['flag_kaggle'] else "⏭️"
                print(f"📁 {data['mfo']}: Kaggle {kaggle_status} | Google {google_status}")
            
            print("="*60)
            
            # Return the last processed link's flags
            last_result = list(results.values())[-1] if results else {'flag_kaggle': False, 'flag_google': False}
            return (last_result['flag_kaggle'], last_result['flag_google'])
        else:
            print("❌ No results to display")
            return (False, False)
            
    except Exception as e:
        print(f"❌ Error in main process: {str(e)}")
        return (False, False)


    print("\n" + "="*60)
    print("🏁 FINAL RESULTS:")
    print(f"📤 Kaggle Flag: {'✅ SUCCESS' if flag_kaggle else '❌ FAILED'}")
    print(f"☁️ Google Flag: {'✅ SUCCESS' if flag_google else '❌ FAILED'}")
    print("="*60)