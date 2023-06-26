import os
import time
import xlrd



def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    dataFile = []
    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    return dataFile


def check_bug():
    exle_file = read_xlrd('../data/WPC/mos.xls')
    datalists = os.listdir('../data/WPC/Distortion_ply')

    temp_list = []
    for file in exle_file:
        temp_list.append(file[0])
        if file[0] not in datalists:
            print(f'bug:    {file[0]}    in mos.xls,     but not in Distortion_ply')

    print(f'\n\n\n\033[1;35mJust for split, if there are no more characters, the files is all renamed and correct.\033[0m\n\n\n')
    for data in datalists:
        if data not in temp_list:
            print(f'bug:    {data}   in Distortion_ply,     but not in mos.xls')


def rename_error_file():
    ply_dir = '../data/WPC/Distortion_ply'
    ply_name_list = os.listdir(ply_dir)
    error_count = 0
    for ply_name in ply_name_list:
        if '_rounded' in ply_name:
            error_count += 1
            temp_name = ply_name.split('_rounded')[0] + '.ply'
            os.rename(os.path.join(ply_dir, ply_name), os.path.join(ply_dir, temp_name))     # rename the file

    print(f'There are {error_count} files need to be renamed.')



if __name__ == '__main__':
    
    rename_error_file()
    check_bug()
    
    start_time = time.time()
    time.sleep(2)
    print(f'{(time.time()-start_time)/3600:.4f} s')
