import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Identify:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data')
        self.modalities = kwargs.get('modalities')
        self.fused_modalities = kwargs.get('fused_modalities')
        self.experiment_id = kwargs.get('exp_id')
        self.k = kwargs.get('k')

        self.hits = pd.DataFrame(columns=self.modalities)

    def pad_labels(self, L):
        n = self.k
        if len(L) < n:
            L.extend([0] * (n - len(L)))
        return L

    def chop_and_sort(self):
        for mod in self.modalities+self.fused_modalities:

            ## Sort the gallerys scores returned for probe
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

    def generate_plots(self):
        #### baseline plots
        for modality in self.modalities:
            hits = self.hits[modality]
            plt.plot([i for i in range(1, self.k + 1)], hits, label=modality, marker='o', linestyle='dotted')

        plt.legend()
        plt.xlabel('Rank')
        plt.ylabel('Identification Rate')
        plt.xticks([i for i in range(1, self.k + 1)])

        plt.title('CMC Curve')
        plt.savefig('./generated/experiments/CMC/' + self.experiment_id + '/CMC-baseline.png')
        plt.clf()

        #### fused plots
        for modality in self.fused_modalities:
            hits = self.hits[modality]
            plt.plot([i for i in range(1, self.k + 1)], hits, label=modality, marker='D', linestyle='dotted')

        plt.legend()
        plt.xlabel('Rank')
        plt.ylabel('Identification Rate')
        plt.xticks([i for i in range(1, self.k + 1)])

        plt.title('CMC Curve')
        plt.savefig('./generated/experiments/CMC/' + self.experiment_id + '/CMC-fused.png')
        plt.clf()

        ## Plot all
        for modality in self.modalities:
            hits = self.hits[modality]
            plt.plot([i for i in range(1, self.k + 1)], hits, label=modality, marker='o', linestyle='dotted')
        for modality in self.fused_modalities:
            hits = self.hits[modality]
            plt.plot([i for i in range(1, self.k + 1)], hits, label=modality, marker='D', linestyle='dotted')

        plt.legend()
        plt.xlabel('Rank')
        plt.ylabel('Identification Rate')
        plt.xticks([i for i in range(1, self.k + 1)])

        plt.title('CMC Curve')
        plt.savefig('./generated/experiments/CMC/' + self.experiment_id + '/CMC-all.png')

    def get_accuracies(self):
        return self.hits