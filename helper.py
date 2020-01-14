import numpy as np
from strlearn.streams import StreamGenerator


def toystreams(random_state):
    # Variables
    distributions = [[0.95, 0.05], [0.90, 0.10]]
    label_noises = [0.01]
    incremental = [False]
    ccs = [5, None]
    n_drifts = 1

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                for spacing in ccs:
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=random_state,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        n_chunks=20,
                        chunk_size=250,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams


def streams(random_state):
    # Variables
    distributions = [[0.95, 0.05], [0.90, 0.10], [0.85, 0.15], [0.80, 0.20]]
    label_noises = [
        0.01,
        0.05,
        0.10,
    ]
    incremental = [False, True]
    ccs = [5, None]
    n_drifts = 1

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                for spacing in ccs:
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=random_state,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        chunk_size=250,
                        n_chunks=200
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams
