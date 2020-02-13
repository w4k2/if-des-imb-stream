import numpy as np

class DriftEvaluator:

    def __init__(self, scores, drift_indices):
        self.__scores = []
        self.__drift_indices= drift_indices

        for pair in scores[0]:
            self.__scores.append(np.mean(pair))

        self.__median_scores = self.calculate_median_scores()
        self.__recovery_lengths = self.calculate_recovery_lengths()
        self.__performance_loss = self.calculate_performance_loss()

    def get_max_performance_loss(self):
        return self.__performance_loss

    def get_recovery_lengths(self):
        return self.__recovery_lengths

    def get_mean_acc(self):
        return np.mean(self.__scores)

    def calculate_median_scores(self):
        median_scores = []
        prev_idx = 0
        for idx in self.__drift_indices:
            median_scores.append(np.median(self.__scores[prev_idx:idx]))
            prev_idx = idx
        median_scores.append(np.median(self.__scores[prev_idx:]))

        return median_scores

    def calculate_recovery_lengths(self):
        recovery_lengths = []
        median_score_idx = 1
        for idx in self.__drift_indices:
            score_idx = idx+1
            while self.__scores[score_idx] < 0.95*self.__median_scores[median_score_idx]:
                score_idx += 1
            recovery_lengths.append((score_idx-idx)/1000) # divided by number of chunks
            median_score_idx += 1

        return recovery_lengths

    def calculate_performance_loss(self):
        performance_loss = []

        stream_a_idx = 0
        stream_b_idx = stream_a_idx + 1

        for idx in self.__drift_indices:
            min_score = min(self.__scores[idx-10:idx+10])
            min_stream_score = min(self.__median_scores[stream_a_idx], self.__median_scores[stream_b_idx])
            loss = (min_stream_score - min_score) / min_stream_score
            performance_loss.append(loss)
            stream_a_idx += 1
            stream_b_idx += 1

        return performance_loss
