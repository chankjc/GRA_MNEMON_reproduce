from tqdm import tqdm


def confusion_matrix(reconstruct_edge, real_edges):
    reconstruct_edge_size = len(reconstruct_edge[0])
    real_edges_size = len(real_edges[0])

    reconstruct_edge_set = set()
    real_edges_set = set()

    for i in tqdm(range(reconstruct_edge_size)):
        if reconstruct_edge[0][i] == reconstruct_edge[1][i]:
            continue
        reconstruct_edge_set.add(
            (int(reconstruct_edge[0][i]), int(reconstruct_edge[1][i]))
        )

    for i in tqdm(range(real_edges_size)):
        if real_edges[0][i] == real_edges[1][i]:
            continue
        real_edges_set.add((int(real_edges[0][i]), int(real_edges[1][i])))

    tp = reconstruct_edge_set.intersection(real_edges_set)
    fp = reconstruct_edge_set - tp
    fn = real_edges_set - tp

    tp = len(tp)
    fp = len(fp)
    fn = len(fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 / ((1 / precision) + (1 / recall))
    tqdm.write("============================")
    tqdm.write(f"> precision: {precision:.2%}")
    tqdm.write(f"> recall: {recall:.2%}")
    tqdm.write(f"> f1_score: {f1_score:.2%}")
    tqdm.write("============================")
    return precision, recall, f1_score
