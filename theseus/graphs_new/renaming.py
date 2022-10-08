import os
import glob


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


origin = os.getcwd()
origin

path_category = glob.glob('HighlyEntangledStates/*/*.json') # this already ignores the folders 'other_solutions'

path_category

for path in sorted(path_category):
    slash1 = find_nth(path,'/',1)
    slash2 = find_nth(path,'/',2)
    folder_name = path[ slash1 +1: slash2]
    total_path = origin + '/' + path 
    if ('/plot_' in path) or ('/config_' in path):
        pass
    else: 
        if ('rough' in path) or ('clean' in path):
            new_path = total_path.replace(folder_name+'/',f'{folder_name}/plot_{folder_name}_')
        else:
            new_path = total_path.replace(folder_name+'/',f'{folder_name}/config_')
        os.rename(total_path,new_path)
