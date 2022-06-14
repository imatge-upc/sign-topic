'''
Ingests csv files with VIDEO ID - TEXT mapping and
prepares data to fit into the data format expected by fairseq's RoBERTa.
'''

import pandas as pd


def main():
    for split in ['train', 'test', 'val']:
        data = pd.read_csv(f'{split}_filt.csv', sep='\t')

        f1 = open(split + '.input0', 'w')
        f2 = open(split + '.label', 'w')
        for _, row in data.iterrows():
            f1.write(row['TEXT'] + '\n')
            f2.write(str(row['CATEGORY_ID'] - 1) + '\n')  # in fairseq, labels must be 0-indexed
        f1.close()
        f2.close()

if __name__=='__main__':
    main()