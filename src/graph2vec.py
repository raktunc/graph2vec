"""Graph2Vec module."""
import os
import glob
import hashlib

import numpy as np
import pandas as pd
import networkx as nx
from numpy.linalg import LinAlgError
from scipy.spatial.distance import euclidean, cosine, minkowski, mahalanobis, braycurtis, canberra, chebyshev, \
    cityblock, correlation
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)

    edgelist = []
    with open(path) as f:
        for line in f:
            edge = line.split()
            if len(edge) > 2:
                del edge[2]
            edgelist.append([int(node) for node in edge])

    graph = nx.from_edgelist(edgelist, nx.DiGraph)

    features = nx.degree(graph)
    features = {int(k): v for k, v in features}

    return graph, features, name


def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc


def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)
        out.append([int(identifier)] + list(model.dv["g_" + identifier]))
    column_names = ["timeStep"] + ["x_" + str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["timeStep"])
    out.to_csv(output_path, index=None)
    out.drop("timeStep", inplace=True, axis=1)
    return out


def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    process_start_time = time.time()

    graphs = glob.glob(os.path.join(args.input_path, "[0-9].txt"))
    graphs.extend(glob.glob(os.path.join(args.input_path, "[0-9][0-9].txt")))

    print("\nFeature extraction started.\n")
    start_time = time.time()
    document_collections = Parallel(n_jobs=args.workers)(
        delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nFeature extraction completed in %s seconds\n" % (time.time() - start_time))

    print("\nOptimization started.\n")
    start_time = time.time()
    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)
    print("\nOptimization completed in %s seconds\n" % (time.time() - start_time))

    vectors = save_embedding(args.output_path + "/vectors.csv", model, graphs, args.dimensions).values.tolist()

    print("\nDistance computation started.\n")
    start_time = time.time()
    initial = np.array(vectors[0])
    del vectors[0]  # delete the initial networks vector
    df = pd.DataFrame(vectors)
    inv_cov = np.linalg.inv(df.cov())
    means = df.mean()
    dists_map_initial = compute_distances(vectors, initial, inv_cov, args.output_path + "/distsToInitial.csv")
    dists_map_mean = compute_distances(vectors, means, inv_cov, args.output_path + "/distsToMean.csv")
    print("\nDistance computation completed in %s seconds\n" % (time.time() - start_time))

    events_map_initial = compute_events(dists_map_initial)
    events_map_mean = compute_events(dists_map_mean)

    compute_average_precisions(events_map_initial, args.ground_truth, args.output_path + "/averagePrecisionsInitial.csv")
    compute_average_precisions(events_map_mean, args.ground_truth, args.output_path + "/averagePrecisionsMean.csv")

    print("\nTotal process completed in %s seconds\n" % (time.time() - process_start_time))


def compute_average_precisions(events_map, ground_truth_events, output_path):
    average_precisions_map = {}
    for key, value in events_map.items():
        average_precisions_map[key] = compute_average_precision(value, ground_truth_events)
    pd.DataFrame(average_precisions_map, index=[0]).to_csv(output_path, index=False)
    #  print(output_path + ":")
    #  print(average_precisions_map)
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
    pd.DataFrame(dists_map).to_csv(output_path, index=False)
    return dists_map


if __name__ == "__main__":
    args = parameter_parser()
    main(args)
