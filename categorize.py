import sys
import os
import shutil
import re

'''
place it inside midis folder. Composer lists are copy and pasted from wikipedia without careful selection. There must be some composers that has been misclassified.
Generates 4 folders each contains different era's composer's work, those who are not in baroque/classic/romantic are moved to contemporary era.
The classification is only based on the era that the composer lived in, not based on the music itself.
'''
def read_files(file_path, baroque_list, classical_list, romantic_list):
    baroque_folder = '.\\baroque'
    classic_folder = '.\\classical'
    romantic_folder = '.\\romantic'
    contemporary_folder = '.\\contemporary'
    composer_set = set()
    for filename in os.listdir(file_path):
        if filename.endswith('mid'):
            tokens = filename.split(',')
            name = tokens[1] + ' ' + tokens[0]
            
            #print(name)
            name = name.lower()[1:]
            
            composer_set.add(name)
            if name in baroque_list:
                #print(name)
                shutil.move(filename, baroque_folder)
            elif name in classical_list:
                #print(name)
                shutil.move(filename, classic_folder)
            elif name in romantic_list:
                shutil.move(filename, romantic_folder)
            else:
                shutil.move(filename, contemporary_folder)
    return composer_set

        
def get_composers_by_era(name_list):
    composers = []
    with open(name_list, 'r', encoding='mac_roman') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(',')[0]
            a = r'\(.*\)'
            b = r'\[.*\]'
            line = re.sub(a, '', line)
            line = re.sub(b, '', line)
            line = line.lower()
            composers.append(line[:-1])
    
    #print(len(baroque_composers))
    return composers


def main():
    midi_path = '.'
    baroque_list_path = '..\\baroque_list'
    classical_list_path = '..\\classic_list'
    romantic_list_path = '..\\romantic_list'
    baroque_composers = get_composers_by_era(baroque_list_path)
    classical_composers = get_composers_by_era(classical_list_path)
    romantic_composers = get_composers_by_era(romantic_list_path)
    composer_set = read_files(midi_path, baroque_composers, classical_composers, romantic_composers)
    

    
if __name__ == "__main__":
    main()