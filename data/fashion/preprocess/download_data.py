import os
import sys

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

os.system('rm -rf ../data')

# destination = 'ditto_femnist.zip'
# os.system('wget https://www.dropbox.com/s/psczerf2oh7geqr/' + destination)

import gdown

url = 'https://drive.google.com/u/0/uc?id=1WyU9IYHUeLPElFuzPc8--qB7dRPuNZBR'

destination = 'fashion.zip'
gdown.download(url, destination, quiet=False)
os.system('mkdir -p ../data')
os.system('unzip -o ' + destination + " -d ../")
os.system('rm ' + destination)
os.system('rm -rf ../data/_*')
os.system('mv -f ../fashion/data/* ../')
