import os
import sys

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

os.system('rm -rf ../data')

# destination = 'ditto_femnist.zip'
# os.system('wget https://www.dropbox.com/s/psczerf2oh7geqr/' + destination)

destination = 'femnist_mini.zip'
os.system('wget https://www.dropbox.com/s/8wi3yw8z8f1apqq/' + destination)
os.system('mkdir -p ../data')
os.system('unzip -o ' + destination + " -d ../")
os.system('rm ' + destination)
os.system('rm -rf ../data/_*')
