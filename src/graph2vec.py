"""Graph2Vec module."""
import glob
import os

import networkx as nx
import numpy as np
import pandas as pd
import time
from karateclub import Graph2Vec, WaveletCharacteristic, LDP, FeatherGraph
from numpy.linalg import LinAlgError
from scipy.spatial.distance import euclidean, cosine, minkowski, mahalanobis, braycurtis, canberra, chebyshev, \
    cityblock, correlation

from param_parser import parameter_parser


def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    process_start_time = time.time()

    graph_files = get_graph_files(args)
    vectors = compute_graph_embeddings(args, graph_files)
    dists_map_initial, dists_map_mean = compute_distances_among_time_steps(args, vectors)
    detect_events_and_evaluate(args, dists_map_initial, dists_map_mean)

    print("\nTotal process completed in %s seconds\n" % (time.time() - process_start_time))


def get_graph_files(args):
    graph_files = glob.glob(os.path.join(args.input_path, "[0-9]SLM.txt"))
    graph_files.extend(glob.glob(os.path.join(args.input_path, "[0-9][0-9]SLM.txt")))
    if len(graph_files) == 0:
        graph_files = glob.glob(os.path.join(args.input_path, "[0-9].txt"))
        graph_files.extend(glob.glob(os.path.join(args.input_path, "[0-9][0-9].txt")))
    return graph_files


def compute_graph_embeddings(args, graph_files):
    print("\nGraph embedding started.\n")
    start_time = time.time()
    graphs = list(map(get_nx_graph_from_file, graph_files))
    graph2vec_model = Graph2Vec()
    graph2vec_model.fit(graphs)
    vectors = graph2vec_model.get_embedding().tolist()
    pd.DataFrame(vectors).to_csv(args.output_path + "/vectors.csv", index=False, header=False)
    print("\nGraph embedding completed in %s seconds\n" % (time.time() - start_time))
    return vectors


def compute_distances_among_time_steps(args, vectors):
    print("\nDistance computation started.\n")
    start_time = time.time()
    initial = np.array(vectors[0])
    del vectors[0]  # delete the initial networks vector
    df = pd.DataFrame(vectors)
    try:
        inv_cov = np.linalg.inv(df.cov())
    except:
        inv_cov = None
    means = df.mean()
    dists_map_initial = compute_distances(vectors, initial, inv_cov, args.output_path + "/distsToInitial.csv")
    dists_map_mean = compute_distances(vectors, means, inv_cov, args.output_path + "/distsToMean.csv")
    print("\nDistance computation completed in %s seconds\n" % (time.time() - start_time))
    return dists_map_initial, dists_map_mean


def detect_events_and_evaluate(args, dists_map_initial, dists_map_mean):
    print("\nEvent detection and evaluation started.\n")
    start_time = time.time()
    events_map_initial = compute_events(dists_map_initial)
    events_map_mean = compute_events(dists_map_mean)
    average_precisions_initial = compute_average_precisions(events_map_initial, args.ground_truth)
    average_precisions_mean = compute_average_precisions(events_map_mean, args.ground_truth)
    pd.DataFrame(merge_dictionary(average_precisions_initial, average_precisions_mean),
                 index=["mean", "initial"]).to_csv(args.output_path + "/averagePrecisions.csv")
    print("\nEvent detection and evaluation completed in %s seconds\n" % (time.time() - start_time))


def get_nx_graph_from_file(path):
    return nx.read_edgelist(path, nodetype=int, data=(("weight", int),), create_using=nx.DiGraph)


def compute_distances(vectors, reference, inv_cov, output_path):
    euclidian_dists = []
    cosine_dists = []
    minkowski_dists = []
    mahalanobis_dists = []
    braycurtis_dists = []
    canberra_dists = []
    chebyshev_dists = []
    cityblock_dists = []
    correlation_dists = []
    i = 0
    while i < len(vectors):
        euclidian_dists.append(round(euclidean(np.array(vectors[i]), reference), 3))
        cosine_dists.append(round(cosine(np.array(vectors[i]), reference), 3))
        minkowski_dists.append(round(minkowski(np.array(vectors[i]), reference, 1), 3))
        if inv_cov is not None:
            mahalanobis_dists.append(round(mahalanobis(np.array(vectors[i]), reference, inv_cov), 3))
        braycurtis_dists.append(round(braycurtis(np.array(vectors[i]), reference), 3))
        canberra_dists.append(round(canberra(np.array(vectors[i]), reference), 3))
        chebyshev_dists.append(round(chebyshev(np.array(vectors[i]), reference), 3))
        cityblock_dists.append(round(cityblock(np.array(vectors[i]), reference), 3))
        correlation_dists.append(round(correlation(np.array(vectors[i]), reference), 3))
        i += 1
    dists_map = {'euclidian': euclidian_dists,
                 'cosine': cosine_dists,
                 'minkowski': minkowski_dists,
                 'mahalanobis': mahalanobis_dists,
                 'braycurtis': braycurtis_dists,
                 'canberra': canberra_dists,
                 'chebyshev': chebyshev_dists,
                 'cityblock': cityblock_dists,
                 'correlation': correlation_dists
                 }
    if inv_cov is None:
        del dists_map['mahalanobis']
    pd.DataFrame(dists_map).to_csv(output_path, index=False)
    return dists_map


def compute_events(dists_map):
    events_map = {}
    for key, value in dists_map.items():
        events_map[key] = detect_events(value)
    return events_map


def detect_events(dists):
    events_map = {}
    min_dist = min(dists)
    max_dist = max(dists)
    if min_dist != max_dist:
        for threshold in np.linspace(min_dist, max_dist, 100):
            events_map[threshold] = []
            for i, dist in enumerate(dists):
                if dist >= threshold:
                    events_map[threshold].append(i + 1)
    return events_map


def compute_average_precisions(events_map, ground_truth_events):
    average_precisions_map = {}
    for key, value in events_map.items():
        average_precisions_map[key] = compute_average_precision(value, ground_truth_events)
    return average_precisions_map


def compute_average_precision(events_map, ground_truth_events):
    recall_precision_map = {}
    for threshold, events in events_map.items():
        detected_true_events = np.intersect1d(events, ground_truth_events)
        precision = 0 if len(events) == 0 else len(detected_true_events) / len(events)
        recall = len(detected_true_events) / len(ground_truth_events)
        recall_precision_map[recall] = precision
    average_precision = 0.0
    prev_recall = 0.0
    prev_precision = 0.0
    for recall, precision in sorted(recall_precision_map.items()):
        average_precision += ((prev_precision + precision) / 2 * (recall - prev_recall))
        prev_recall = recall
        prev_precision = precision
    return round(average_precision, 2)


def merge_dictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = [value, dict_1[key]]
    return dict_3


if __name__ == "__main__":
    args = parameter_parser()
    main(args)
