import numpy as np
from strlearn.streams import StreamGenerator, ARFFParser


def realstreams():
    return {
        "covtypeNorm-1-2vsAll": ARFFParser("datasets/covtypeNorm-1-2vsAll-pruned.arff", n_chunks=265, chunk_size=1000),
        "poker-lsn-1-2vsAll": ARFFParser("datasets/poker-lsn-1-2vsAll-pruned.arff", n_chunks=359, chunk_size=1000),
    }

def realstreams2():
    return {
        "covtypeNorm-1-2vsAll": ARFFParser("datasets/covtypeNorm-1-2vsAll-pruned.arff", n_chunks=265, chunk_size=1000),
        "poker-lsn-1-2vsAll": ARFFParser("datasets/poker-lsn-1-2vsAll-pruned.arff", n_chunks=359, chunk_size=1000),
    }

def moa_streams():
    return {
        "gr_css5_rs804_nd1_ln1_d85_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln1_d85_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln1_d90_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln1_d90_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln1_d95_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln1_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln3_d85_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln3_d85_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln3_d90_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln3_d90_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln3_d95_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln3_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln5_d85_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln5_d85_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln5_d90_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln5_d90_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln5_d95_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln5_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln1_d85_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln1_d85_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln1_d90_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln1_d90_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln1_d95_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln1_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln3_d85_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln3_d85_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln3_d90_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln3_d90_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln3_d95_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln3_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln5_d85_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln5_d85_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln5_d90_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln5_d90_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln5_d95_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln5_d95_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln1_d85_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln1_d85_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln1_d90_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln1_d90_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln1_d95_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln1_d95_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln3_d85_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln3_d85_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln3_d90_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln3_d90_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln3_d95_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln3_d95_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln5_d85_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln5_d85_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln5_d90_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln5_d90_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln5_d95_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln5_d95_50000.arff", n_chunks=200, chunk_size=250),
    }


def toystreams(random_state):
    # Variables
    distributions = [[0.95, 0.05], [0.90, 0.10]]
    label_noises = [
        0.01,
        0.03,
        0.05,
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
                        n_chunks=200,
                        n_clusters_per_class=1,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams


def streams(random_state):
    # Variables
    distributions = [[0.95, 0.05], [0.90, 0.10], [0.85, 0.15]]
    label_noises = [
        0.01,
        0.03,
        0.05,
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
                        n_chunks=200,
                        n_clusters_per_class=1,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams


def timestream(chunk_size):
    # Variables
    distributions = [[0.80, 0.20]]
    label_noises = [
        0.01,
    ]
    incremental = [False]
    ccs = [None]
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
                        random_state=1994,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        chunk_size=chunk_size,
                        n_chunks=5,
                        n_clusters_per_class=1,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams
