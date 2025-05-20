import os
import time
import math
import numpy as np
from scipy.sparse import csc_matrix, diags
import warnings
from scipy.sparse import SparseEfficiencyWarning

# Ignore SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

print(1)


def compute_personalization_vector(max_value):
    """
    Computes the initial personalization vector p0.
    """
    return (1 / float(max_value + 1)) * np.ones((max_value + 1, 1))


def compute_zero_transpose(GIPV, zero_indices, max_value):
    """
    Computes the Zero transpose vector Zt.
    """
    WS = sum(GIPV[zero_indices]) / float(max_value + 1)
    Zt = np.full((max_value + 1, 1), WS)
    return Zt


def generalized_page_rank(O, p0, zero_indices, max_value, alpha, setting_time):
    """
    Performs the Generalized PageRank (GPR) algorithm.
    """
    start_time = time.time()
    GIPV = p0.copy()
    Zt = compute_zero_transpose(GIPV, zero_indices, max_value)

    iteration = 0
    while True:
        iteration += 1
        GIPV_prev = GIPV.copy()
        GIPV = (1 - alpha) * O.dot(GIPV_prev) + (1 - alpha) * Zt + alpha * p0

        error = np.linalg.norm(GIPV - GIPV_prev, ord=1)
        if error < 1e-8:
            break

        Zt = compute_zero_transpose(GIPV, zero_indices, max_value)

    np.savetxt("./GPR.txt", np.column_stack((np.arange(1, max_value + 2), GIPV)), fmt="%d %f")

    total_time = time.time() - start_time + setting_time

    for method in range(3):
        retrieval_start = time.time()
        make_tree_eval_file_for_gpr("./data/indri-lists", GIPV, method)
        retrieval_time = time.time() - retrieval_start
        method_name = ["GPR-NS", "GPR-WS", "GPR-CM"][method]
        print(f"{method_name}: {total_time:.2f} secs for PageRank, {retrieval_time:.2f} secs for retrieval")


def topic_sensitive_page_rank(O, p0, zero_indices, max_value, num_of_topics, num_of_users,
                              dir_doc_topics, dir_query_topic_distro, dir_user_topic_distro,
                              dir_indri_lists, setting_time, alpha=0.8, beta=0.1, gamma=0.1):
    """
    Performs the Topic-Sensitive PageRank (TSPR) algorithm.
    """
    for input_dir, state_name in [(dir_query_topic_distro, "QTSPR"), (dir_user_topic_distro, "PTSPR")]:
        start_time = time.time()
        p_matrix = create_p_matrix(dir_doc_topics, max_value, num_of_topics)
        IPV = np.tile(p0, (1, num_of_topics))

        IPV = compute_topic_pagerank(O, IPV, p0, p_matrix, zero_indices, max_value, alpha, beta, gamma)

        user_ids, query_ids, pr = read_topic_distributions(input_dir, num_of_topics)

        user_id = 2
        query_id = 1
        rq = define_rq(max_value, num_of_topics, pr, IPV, user_ids, query_ids, user_id, query_id)
        np.savetxt(f"./{state_name}-U{user_id}Q{query_id}.txt", np.column_stack((np.arange(1, max_value + 2), rq)), fmt="%d %f")

        total_time = time.time() - start_time + setting_time

        for method in range(3):
            retrieval_start = time.time()
            make_tree_eval_file_for_tspr(dir_indri_lists, pr, IPV,
                                         state=0 if input_dir == dir_query_topic_distro else 1,
                                         method=method, user_ids=user_ids, query_ids=query_ids)
            retrieval_time = time.time() - retrieval_start
            method_name = ["NS", "WS", "CM"][method]
            print(f"{state_name}-{method_name}: {total_time:.2f} secs for PageRank, {retrieval_time:.2f} secs for retrieval")


def create_p_matrix(dir_doc_topics, max_value, num_of_topics):
    """
    Creates the topic distribution matrix 'p' for documents.
    """
    doc_topics = np.loadtxt(dir_doc_topics, dtype=int) - 1  # Adjusting for 0-based indexing
    doc_indices = doc_topics[:, 0]
    topic_indices = doc_topics[:, 1]

    p = np.zeros((max_value + 1, num_of_topics))
    for topic in range(num_of_topics):
        topic_docs = doc_indices[topic_indices == topic]
        if len(topic_docs) > 0:
            p[topic_docs, topic] = 1 / len(topic_docs)
    return p


def compute_topic_pagerank(O, IPV, p0, p_matrix, zero_indices, max_value, alpha, beta, gamma):
    """
    Computes topic-specific PageRank vectors.
    """
    num_of_topics = IPV.shape[1]
    Zt = compute_zero_transpose(IPV[:, 0], zero_indices, max_value)

    for topic in range(num_of_topics):
        iteration = 0
        while True:
            iteration += 1
            IPV_prev = IPV[:, topic].copy()
            IPV[:, topic] = (alpha * O.dot(IPV_prev) + alpha * Zt.flatten() +
                             beta * p_matrix[:, topic] + gamma * p0.flatten())

            error = np.linalg.norm(IPV[:, topic] - IPV_prev, ord=1)
            if error < 1e-8:
                break

            Zt = compute_zero_transpose(IPV[:, topic], zero_indices, max_value)
    return IPV


def read_topic_distributions(input_dir, num_of_topics):
    """
    Reads topic distributions from the given file.
    Expected format per line:
    user_id query_id topic_id1:prob1 topic_id2:prob2 ... topic_idN:probN
    """
    user_ids = []
    query_ids = []
    topic_distributions = []

    with open(input_dir, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            query_id = int(parts[1])
            topic_probs = []
            for item in parts[2:]:
                if ':' in item:
                    _, prob_str = item.split(':', 1)
                else:
                    prob_str = item 
                try:
                    prob = float(prob_str)
                except ValueError:
                    prob = 0.0
                topic_probs.append(prob)
            if len(topic_probs) < num_of_topics:
                topic_probs += [0.0] * (num_of_topics - len(topic_probs))
            elif len(topic_probs) > num_of_topics:
                topic_probs = topic_probs[:num_of_topics]
            user_ids.append(user_id)
            query_ids.append(query_id)
            topic_distributions.append(topic_probs)
    pr = np.array(topic_distributions)
    return user_ids, query_ids, pr


def define_rq(max_value, num_of_topics, pr, IPV, user_ids, query_ids, user_id, query_id):
    """
    Defines the Rq vector for a specific user and query.
    """
    for idx, (uid, qid) in enumerate(zip(user_ids, query_ids)):
        if uid == user_id and qid == query_id:
            rq = (pr[idx, :] @ IPV.T).reshape(-1, 1)
            return rq
    return np.zeros((max_value + 1, 1))


def make_tree_eval_file_for_gpr(indri_lists_dir, IPV, method):
    """
    Generates TreeEval files for GPR algorithm by processing each indri-list file directly.
    """
    method_names = ["GPR-NS", "GPR-WS", "GPR-CM"]
    output_file = f"./{method_names[method]}.txt"

    with open(output_file, 'w') as f_out:
        for filename in os.listdir(indri_lists_dir):
            if filename.endswith(".txt"):
                qid = filename.split('.')[0]
                file_path = os.path.join(indri_lists_dir, filename)
                indices = []
                scores = []
                with open(file_path, 'r') as f_in:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue 
                        doc_id = int(parts[2])
                        try:
                            indri_score = float(parts[4])
                        except ValueError:
                            indri_score = 0.0
                        indices.append(doc_id - 1)
                        scores.append(indri_score)

                temp_scores = []
                for idx_doc, doc_index in enumerate(indices):
                    if method == 0:  
                        temp_score = IPV[doc_index, 0]
                    elif method == 1: 
                        temp_score = 0.9 * IPV[doc_index, 0] + 0.1 * scores[idx_doc]
                    else:  
                        if IPV[doc_index, 0] > 0:
                            temp_score = 0.5 * math.log(IPV[doc_index, 0]) + 0.5 * scores[idx_doc]
                        else:
                            temp_score = 0.5 * math.log(1e-10) + 0.5 * scores[idx_doc]  # Use a small value
                    temp_scores.append((doc_index + 1, temp_score))

                temp_scores.sort(key=lambda x: x[1], reverse=True)

                for rank, (doc_id, score) in enumerate(temp_scores, start=1):
                    f_out.write(f"{qid} Q0 {doc_id} {rank} {score} run-1\n")
    print(f"Finished creating {output_file}")


def make_tree_eval_file_for_tspr(indri_lists_dir, pr, IPV, state, method, user_ids, query_ids):
    """
    Generates TreeEval files for TSPR algorithm by processing each indri-list file directly.
    """
    method_names = ["NS", "WS", "CM"]
    state_names = ["QTSPR", "PTSPR"]
    output_file = f"./{state_names[state]}-{method_names[method]}.txt"

    with open(output_file, 'w') as f_out:
        for filename in os.listdir(indri_lists_dir):
            if filename.endswith(".txt"):
                qid = filename.split('.')[0]
                if '-' in qid:
                    user_id_str, query_id_str = qid.split('-')
                    user_id = int(user_id_str)
                    query_id = int(query_id_str)
                else:
                    continue

                match_indices = [idx for idx, (uid, qid_val) in enumerate(zip(user_ids, query_ids))
                                 if uid == user_id and qid_val == query_id]
                if not match_indices:
                    continue
                idx = match_indices[0]
                pr_vector = pr[idx]

                rq = (pr_vector @ IPV.T).reshape(-1)

                file_path = os.path.join(indri_lists_dir, filename)
                indices = []
                scores = []
                with open(file_path, 'r') as f_in:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        doc_id = int(parts[2])
                        try:
                            indri_score = float(parts[4])
                        except ValueError:
                            indri_score = 0.0
                        indices.append(doc_id - 1)
                        scores.append(indri_score)

                temp_scores = []
                for idx_doc, doc_index in enumerate(indices):
                    if method == 0:  # NS
                        temp_score = rq[doc_index]
                    elif method == 1:  # WS
                        temp_score = 0.9 * rq[doc_index] + 0.1 * scores[idx_doc]
                    else:  # CM
                        if rq[doc_index] > 0:
                            temp_score = 0.5 * math.log(rq[doc_index]) + 0.5 * scores[idx_doc]
                        else:
                            temp_score = 0.5 * math.log(1e-10) + 0.5 * scores[idx_doc]  # Use a small value
                    temp_scores.append((doc_index + 1, temp_score))

                # Sort documents by score
                temp_scores.sort(key=lambda x: x[1], reverse=True)

                # Write results
                for rank, (doc_id, score) in enumerate(temp_scores, start=1):
                    f_out.write(f"{qid} Q0 {doc_id} {rank} {score} run-1\n")
    print(f"Finished creating {output_file}")


def main():
    # Directory settings
    dir_doc_topics = "./data/doc_topics.txt"
    dir_query_topic_distro = "./data/query-topic-distro.txt"
    dir_transition = "./data/transition.txt"
    dir_user_topic_distro = "./data/user-topic-distro.txt"
    dir_indri_lists = "./data/indri-lists"

    # Initial settings
    start_setting = time.time()

    doc_topics = np.loadtxt(dir_doc_topics, dtype=int)

    if doc_topics.ndim == 1:
        num_of_topics = 1
    else:
        num_of_topics = doc_topics[:, 1].max() + 1  # +1 for 0-based indexing

    user_ids_q, _, _ = read_topic_distributions(dir_query_topic_distro, num_of_topics)
    num_of_users = max(user_ids_q)

    transition_data = np.loadtxt(dir_transition)

    if transition_data.ndim == 1:
        transition_data = transition_data.reshape(1, -1)

    T_row = transition_data[:, 0].astype(int) - 1  # 0-based indexing
    T_col = transition_data[:, 1].astype(int) - 1  # 0-based indexing
    data = transition_data[:, 2]

    max_value = max(T_row.max(), T_col.max())
    O = csc_matrix((data, (T_col, T_row)), shape=(max_value + 1, max_value + 1))

    # Normalize transition matrix
    ni = np.array(O.sum(axis=0)).flatten()
    scale = np.divide(1.0, ni, out=np.zeros_like(ni), where=ni != 0)
    D = diags(scale)
    O = O.dot(D)

    zero_indices = np.where(ni == 0)[0]

    # Personalization vector
    p0 = compute_personalization_vector(max_value)

    setting_time = time.time() - start_setting

    # Run GPR
    generalized_page_rank(O, p0, zero_indices, max_value, alpha=0.2, setting_time=setting_time)

    # Run TSPR
    topic_sensitive_page_rank(
        O, p0, zero_indices, max_value, num_of_topics, num_of_users,
        dir_doc_topics, dir_query_topic_distro, dir_user_topic_distro,
        dir_indri_lists, setting_time
    )


if __name__ == '__main__':
    main()
