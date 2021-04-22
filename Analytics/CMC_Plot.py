import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CMC:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data')
        self.modalities = kwargs.get('modalities')
        print(self.modalities)
        self.k = kwargs.get('k')

        self.hits = pd.DataFrame(columns=self.modalities)

    def pad_labels(self, L):
        n = self.k
        if len(L) < n:
            L.extend([0] * (n - len(L)))
        return L

    def chop_and_sort(self):
        print('Starting - may god have mercy on our soul')
        for mod in self.modalities:

            ## Sort the galleries returned for probe
            trimmed_sorted = self.data.groupby('Probe_ID').apply(
                lambda x: x.sort_values(mod, ascending=False)).reset_index(drop=True)

            ## drop extra scores (scores lower than rank k)
            trimmed_sorted = trimmed_sorted.groupby('Probe_ID').head(self.k)

            ## This adds 0 labels to transactions who have fewer than k scores
            df = trimmed_sorted.groupby('Probe_ID')['Label'].apply(list).apply(self.pad_labels)
            labels = pd.DataFrame.from_dict(dict(zip(df.index, df.values))).transpose()

            hit_rates = []
            for i in range(1, self.k + 1):
                hits = np.where(labels.iloc[:, :i].sum(axis=1) >= 1.0)
                rate = hits[0].shape[0] / len(labels.index)
                hit_rates.append(rate)

            self.hits[mod] = hit_rates
        self.hits.to_csv('./hit_rates.csv')
        print('SAVED')


    def plots(self):
        print('PLOTTING')
        for modality in self.modalities:
            hits = self.hits[modality]
            if ':' in modality:
                plt.plot([i for i in range(1, self.k + 1)], hits, label=modality, marker='D', linestyle='dotted')

            else:
                plt.plot([i for i in range(1, self.k + 1)], hits, label=modality, marker='o', linestyle='dotted')

        plt.legend()
        plt.xlabel('Rank')
        plt.ylabel('Identification Rate')
        plt.xticks([i for i in range(1, self.k + 1)])

        plt.title('CMC Curve')
        plt.savefig('./CMC.png')