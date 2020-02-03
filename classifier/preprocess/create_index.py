import os
import pandas as pd

allowed_exts = {'mp3', 'wav', 'au'}
labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
folder_csv="data/index"
column_names = ['id', 'filepath', 'label']
folder_dataset_gtg = 'data/genres/'
csv_filename = 'gtzan_genre.csv'

def get_rows_from_folders(folder_dataset, folders):
    rows = []
    for label_idx, folder in enumerate(folders):
        files = os.listdir(os.path.join(folder_dataset, folder))
        files = [f for f in files if f.split('.')[-1].lower() in allowed_exts]
        for fname in files:
            file_path = os.path.join(folder_dataset, folder, fname)
            file_id =fname.split('.')[0]
            file_label = label_idx
            rows.append([file_id, file_path, file_label])
    print('Done - length:{}'.format(len(rows)))
    print(rows[0])
    print(rows[-1])
    return rows


def write_to_csv(rows, column_names, csv_fname):
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(os.path.join(folder_csv, csv_fname))


if __name__ == "__main__":
    if not os.path.exists(folder_csv):
        os.mkdir(folder_csv)

    n_label_gtg = len(labels)
    folders_gtg = [s + '/' for s in labels]

    rows_gtg = get_rows_from_folders(folder_dataset_gtg, folders_gtg)

    write_to_csv(rows_gtg, column_names, csv_filename)
