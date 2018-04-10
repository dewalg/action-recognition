import argparse


def main(source_csv, output_csv, input_txt):
    with open(input_txt) as f:
        ids = f.readlines()
    ids = [line.strip() for line in ids]
    print len(ids)
    output = []
    with open(source_csv) as source:
        csv_lines = source.readlines()
    for csv_line in csv_lines:
        youtube_id = csv_line.split(',')[1]
        if youtube_id == "youtube_id" or youtube_id in ids:
            output.append(csv_line)
    with open(output_csv, 'w') as destination:
        for output_line in output:
            destination.write(output_line)


if __name__ == '__main__':
    description = 'Helper script for converting txt to csv'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('source_csv', type=str,
                   help=('Source CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_csv', type=str,
                   help='Output CSV for saving the file')
    p.add_argument('input_txt', type=str,
                   help='Input TXT for conversion')
    main(**vars(p.parse_args()))
