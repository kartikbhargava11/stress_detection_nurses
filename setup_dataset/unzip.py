import os
import shutil
import multiprocessing

MAIN_PATH = '/Users/kartikbhargava/coding/CE888/'
stress_data_path = MAIN_PATH + 'Stress_dataset'
def unzip_parallel(file, sub_file):
    shutil.unpack_archive(
        os.path.join(stress_data_path, file, sub_file),
        os.path.join(stress_data_path, file, sub_file[:-4])
    )
    print(f'Task {file} executing with {sub_file}', flush=True)

if __name__ == '__main__':
    # shutil.unpack_archive(MAIN_PATH + 'Data.zip', MAIN_PATH + 'Data')
    shutil.unpack_archive(MAIN_PATH + 'Stress_dataset.zip', MAIN_PATH + 'Stress_dataset')

    cpu_count = int(multiprocessing.cpu_count()/2)
    print(f'Using {cpu_count} CPUs')

    new_list = [
        (file, sub_file)
        for file in os.listdir(stress_data_path)
        for sub_file in os.listdir(os.path.join(stress_data_path, file))
    ]

    pool = multiprocessing.Pool(cpu_count)
    results = pool.starmap(unzip_parallel, new_list)
    pool.close()