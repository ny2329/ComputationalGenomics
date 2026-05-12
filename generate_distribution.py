import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METADATA_FILE = 'GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt'

def main():
    df = pd.read_csv(METADATA_FILE, sep='\t', usecols=['SAMPID', 'SMTSD'], low_memory=False)
    brain = df[df['SMTSD'].str.startswith('Brain -', na=False)]
    counts = brain['SMTSD'].value_counts()
    top_n = 6
    top = counts.nlargest(top_n).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9,5))
    colors = sns.color_palette('viridis_r', n_colors=len(top))
    y = range(len(top))
    ax.barh(y, top.values, color=colors, height=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Tissue Region')
    ax.set_title('Distribution of Brain Tissue Samples (GTEx v11)')
    ax.grid(False)
    maxv = top.max()
    ax.set_xlim(0, int(maxv * 1.05))
    step = int(max(1, round(maxv / 4)))
    ax.set_xticks(range(0, int(maxv + step), step))
    fig.tight_layout()
    fig.savefig('expression_heatmap_sample_distribution.png', dpi=200)
    fig.savefig('expression_heatmap_sample_distribution.svg')
    top.to_csv('expression_heatmap_sample_distribution_counts.csv')

if __name__ == '__main__':
    main()
