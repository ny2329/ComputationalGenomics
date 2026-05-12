import pandas as pd
import argparse

METADATA_FILE = 'GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain-only', action='store_true', help='Only include samples with SMTSD starting with "Brain -"')
    parser.add_argument('--out', default='expression_heatmap_onehot.csv')
    args = parser.parse_args()

    df = pd.read_csv(METADATA_FILE, sep='\t', usecols=['SAMPID', 'SMTSD'], low_memory=False)
    if args.brain_only:
        df = df[df['SMTSD'].str.startswith('Brain -', na=False)]

    df = df.dropna(subset=['SMTSD'])
    df = df.set_index('SAMPID')
    onehot = pd.get_dummies(df['SMTSD'], prefix='region')
    onehot.to_csv(args.out)

    # save mapping of original region to onehot column
    mapping = pd.DataFrame({'region': onehot.columns})
    mapping['original'] = mapping['region'].str.replace('region_', '')
    mapping.to_csv('expression_heatmap_onehot_mapping.csv', index=False)
    print(f"Saved one-hot to {args.out} and mapping to expression_heatmap_onehot_mapping.csv")

if __name__ == '__main__':
    main()
