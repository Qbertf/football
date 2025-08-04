import os
import pandas as pd
import subprocess
import json
from pathlib import Path
import glob

def unscramble_json(scrambled_bytes, password):
    """Unscramble data using XOR with password"""
    password_bytes = str(password).encode('utf-8')
    
    unscrambled = bytearray()
    for i in range(len(scrambled_bytes)):
        unscrambled.append(scrambled_bytes[i] ^ password_bytes[i % len(password_bytes)])
    
    return json.loads(unscrambled.decode('utf-8'))

def clone_repos_from_csv(csv_file,types):
    try:
        # Read CSV with pandas
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_columns = {'repo_url', 'branch_name', 'clone_dir'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain: {required_columns}")

        # Clone each repository
        for _, row in df.iterrows():
            repo_url = row['repo_url']
            branch_name = row['branch_name']
            clone_dir = row['clone_dir']
            if row['types']==types:
              os.system('rm -rf '+clone_dir)
              print(f"Cloning {repo_url} (branch: {branch_name}) into {clone_dir}...")
              
              try:
                  cmd = f"git clone --branch {branch_name} --single-branch {repo_url} {clone_dir}"
                  result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
                  print(f"Successfully cloned {repo_url}")
              except subprocess.CalledProcessError as e:
                  print(f"Failed to clone {repo_url}. Error:\n{e.stderr}")

    except Exception as e:
        print(f"Error: {str(e)}")

def firstcheck():
  clone_repos_from_csv("football/infogit.csv","CLI")
  clone_repos_from_csv("football/infogit.csv","MAIN")

  infos=[];
  sessionfiles = glob.glob('CLI/sess*.json')
  for sesfile in sessionfiles:
  
    with Path(sesfile).open("rb") as f:
        loaded_data = f.read()
  
    original_data = unscramble_json(loaded_data, "armeji")
  
    for textbox in original_data:
      running_status = textbox['running_status']
      All_Info = json.loads(textbox['all_info'])
      Machine_Details = eval(All_Info['Machine Details'].replace('^',','))
      RCV_machine = All_Info['Machine Name']
      sessid = All_Info['Session ID']
      sesspath = 'MAIN/'+sessid+'.ses'
      new_status=0;
  
      if os.path.exists(sesspath):
        with open(sesspath,'r') as f:
          new_status = int(f.readlines()[0])
    
      if RCV_machine not in MACHINE_SERVER:
        continue
  
      if running_status!=0 or new_status!=0:
        continue
  
      print('############> valid ',sesspath,new_status,RCV_machine,Machine_Details)
  
      infos.append({'sessid':sessid,'info_machine':Machine_Details,'Raw':All_Info})
  
      return infos
      
import pandas as pd
import os
import shutil
import subprocess

def push_main_from_csv(Machine_Details, source_path, mybranch='status'):
    
    flag = 0  # پیش‌فرض: ناموفق
    base_dir = os.getcwd()
    try:
        
        os.chdir('tempgitkg')
    
        account = Machine_Details['Account']
        repo = Machine_Details['Repo']
        email = Machine_Details['Email']
        token = Machine_Details['Token']
        branch = mybranch

        repo_url = f"https://{token}@github.com/{account}/{repo}.git"
        repo_dir = f"{repo}_{branch}_branch"
        
        if not os.path.exists(repo_dir):
            os.system(f"git clone --branch {branch} --single-branch {repo_url} {repo_dir}")
        else:
            os.system("rm -rf "+repo_dir)
            os.system(f"git clone --branch {branch} --single-branch {repo_url} {repo_dir}")

        os.chdir(repo_dir)

        os.system(f"git config user.email \"{email}\"")
        os.system(f"git config user.name \"{account}\"")
        os.system(f"git checkout {branch}")

        os.system("cp "+source_path+ " .")
        
        try:
            os.system("git add .")
            commit_msg = f"Push {source_path} to branch {branch}"
            
            # تلاش برای commit (اگر چیزی برای کامیت وجود نداشته باشد، خطا نمی‌دهد)
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True,
                text=True
            )
            
            # ادامه بده حتی اگه چیزی برای commit نبود
            push_result = subprocess.run(
                ["git", "push", "origin", branch],
                capture_output=True,
                text=True
            )
            
            if push_result.returncode == 0:
                # عملیات push موفق بوده
                flag = 1
                print("✅ Push succeeded.")
            else:
                # push موفق نبوده
                print("❌ Push failed.")
                print(push_result.stderr)
        
        except Exception as e:
            print("⚠️ Exception during push:", e)
        
        finally:
            os.chdir('../')
        
        print("Flag:", flag)

        os.chdir('../')
    except Exception as e:
        print("⚠️ Exception in main function:", e)
        pass
        
    os.chdir(base_dir)

    return flag
def push_data_from_csv(Machine_Details, source_path):
    
    flag = 0  # پیش‌فرض: ناموفق
    base_dir = os.getcwd()
    try:
        
        os.chdir('tempgitkg')
    
        account = Machine_Details['Account']
        repo = Machine_Details['Repo']
        email = Machine_Details['Email']
        token = Machine_Details['Token']
        branch = Machine_Details['branch']

        repo_url = f"https://{token}@github.com/{account}/{repo}.git"
        repo_dir = f"{repo}_{branch}_branch"
        
        if not os.path.exists(repo_dir):
            os.system(f"git clone --branch {branch} --single-branch {repo_url} {repo_dir}")
        else:
            # اگر ریپو قبلاً وجود داشت، برو داخلش و برنچ مورد نظر رو چک‌اوت و پول کن
            current_dir = os.getcwd()
            os.chdir(repo_dir)
            
            # تغییر به برنچ مورد نظر
            os.system(f"git fetch origin {branch}")
            os.system(f"git checkout {branch}")
            os.system(f"git pull origin {branch}")
            os.chdir(current_dir)

        os.chdir(repo_dir)

        os.system(f"git config user.email \"{email}\"")
        os.system(f"git config user.name \"{account}\"")
        os.system(f"git checkout {branch}")

        os.system("cp -r "+source_path+ " .")
        
        try:
            os.system("git add .")
            commit_msg = f"Push {source_path} to branch {branch}"
            
            # تلاش برای commit (اگر چیزی برای کامیت وجود نداشته باشد، خطا نمی‌دهد)
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True,
                text=True
            )
            
            # ادامه بده حتی اگه چیزی برای commit نبود
            push_result = subprocess.run(
                ["git", "push", "origin", branch],
                capture_output=True,
                text=True
            )
            
            if push_result.returncode == 0:
                # عملیات push موفق بوده
                flag = 1
                print("✅ Push succeeded.")
            else:
                # push موفق نبوده
                print("❌ Push failed.")
                print(push_result.stderr)
        
        except Exception as e:
            print("⚠️ Exception during push:", e)
        
        finally:
            os.chdir('../')
        
        print("Flag:", flag)

        os.chdir('../')
    except Exception as e:
        print("⚠️ Exception in main function:", e)
        pass
        
    os.chdir(base_dir)

    return flag
