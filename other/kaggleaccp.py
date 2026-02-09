import json
import glob
import os
import subprocess
import time
profielslist = glob.glob('kaggleacc/Kaggle/Profile*')

print('profielslist',profielslist)
profiles={}
for p in profielslist:
    profiles.update({p.split('/')[-1].split('_')[0]:p.replace('\\','/')})

print('profiles',profiles)
def init(path):

    print('path',path)

    if path=='':
      return 0
      
    query_profile = path.split('/')[-1].replace('_','')
    
    print('query_profile',query_profile)
    
    with open(profiles[query_profile]+'/kaggle.json', 'r') as f:
        kaggle_api_key = json.load(f)
    
    os.environ['KAGGLE_USERNAME'] = kaggle_api_key['username']
    os.environ['KAGGLE_KEY'] = kaggle_api_key['key']

    print('kaggle_api_key',kaggle_api_key)

    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi


    api = KaggleApi()
    api.authenticate()
    
    # دریافت لیست kernels
    kernels = api.kernels_list(mine=True)
    print('kernels',kernels)

    if 'track' in path:
      kernel_path =  profiles[query_profile] + '/notebook/track'
    else:
      kernel_path =  profiles[query_profile] + '/notebook/calib'

    print('kernel_path',kernel_path)

    result = subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', kernel_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        errors='replace',
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('❌ kernel push failed')

    print('✅ kernel pushed successfully')
    time.sleep(60000)
    print('-----------------------------------------')


    print('-----------------------------------------')

    

