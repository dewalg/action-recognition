import numpy as np
import pandas as pd
import argparse


def parse_kinetics_annotations(input_csv):
    """Returns a parsed DataFrame.

    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    # df.rename(columns={'youtube_id': 'video-id',
    #                    'time_start': 'start-time',
    #                    'time_end': 'end-time',
    #                    'label': 'label-name',
    #                    'is_cc': 'is-cc'}, inplace=True)
    return df

def main(input_csv, output_csv, num_cat, num_samples):
    '''
    main creates a sample of data from the input_csv by selecting
    `num_cat` categories with `num_samples` amount of videos each.
    '''
    # print (input_csv + " " + output_csv + " " + str(num_cat) + " " + str(num_samples))

    data = parse_kinetics_annotations(input_csv)
    all_labels =  list(data['label'].unique())
    num_labels = len(all_labels)
    rand_labels_idx = np.random.choice(num_labels, num_cat)
    rand_labels = [lab for i, lab in enumerate(all_labels) if i in rand_labels_idx]

    out_df = pd.DataFrame(columns=('label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc'))
    for lab in rand_labels:
        lab_df = data.loc[data['label'] == lab]

        num = num_samples
        if num_samples > len(lab_df):
            print("WARNING: number of samples for category {} needed is greater than data available                    - maxing out sample size".format(lab))
            num = len(lab_df)

        print ('adding label ' + lab + ' with ' + str(num) + ' items')
        subsampled_df = lab_df.sample(num)
        out_df = out_df.append(subsampled_df)

    out_df[['time_start', 'time_end', 'is_cc']] = out_df[['time_start', 'time_end', 'is_cc']].astype(float).astype(int)
    out_df.to_csv(output_csv, encoding='utf-8', index=False)


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-i', '--input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('-o', '--output-csv', type=str, default='kinetics_subsample.csv',
                   help='Output csv where sampled video data is stored.')
    p.add_argument('-c', '--num-cat', type=int, default=10)
    p.add_argument('-n', '--num-samples', type=int, default=50)
    main(**vars(p.parse_args()))
