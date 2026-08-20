import torch

def calculate_hamming(B1, B2):
    """
    :param B1:  matrix [m, n]
    :param B2:  matrix [r, n]
    :return: hamming distance [m, r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - torch.matmul(B1, B2.t()))
    return distH

# 现在根据余弦距离设计calculate_cosine函数
def calculate_cosine(B1, B2):
    """
    :param B1:  matrix [m, n]
    :param B2:  matrix [r, n]
    :return: cosine distance [m, r]
    """
    norm_B1 = torch.norm(B1, dim=1, keepdim=True)
    norm_B2 = torch.norm(B2, dim=1, keepdim=True)
    cosine_similarity = torch.matmul(B1, B2.t()) / (norm_B1 * norm_B2.t())
    return 1 - cosine_similarity  # Cosine distance is 1 - cosine similarity

# 根据欧式距离设计calculate_euclidean函数
def calculate_euclidean(B1, B2):
    """
    :param B1:  matrix [m, n]
    :param B2:  matrix [r, n]
    :return: euclidean distance [m, r]
    """
    diff = B1.unsqueeze(1) - B2.unsqueeze(0)  # [m, r, n]
    euclidean_dist = torch.norm(diff, dim=2)  # [m, r]
    return euclidean_dist


def normalize_similarity(distances):
    """
    :param distances: array of hamming distances [k]
    :return: normalized similarity scores [k]
    """
    max_dist = torch.max(distances, dim=1, keepdim=True)[0]
    min_dist = torch.min(distances, dim=1, keepdim=True)[0]
    if torch.all(max_dist == min_dist):
        return torch.ones_like(distances) / distances.shape[1]
    else:
        return (max_dist - distances) / (max_dist - min_dist + 0.001)

def find_topk_similar_hashes_batch(query_hashes, database_hashes, k):
    """
    :param query_hashes: {-1,+1}^{M*q} batch of query hash codes
    :param database_hashes: {-1,+1}^{N*q} database hash codes
    :param k: number of top similar hashes to return
    :return: top_k_indices [M, k], top_k_weights [M, k]
    """
    # Calculate the Hamming distances between the query hashes and all database hashes
    hamming_distances = calculate_hamming(query_hashes, database_hashes)
    
    # Get the indices of the top k smallest Hamming distances
    top_k_indices = torch.topk(hamming_distances, k, largest=False, dim=1).indices
    
    # Get the top k distances
    top_k_distances = torch.gather(hamming_distances, 1, top_k_indices)
    
    # Normalize the distances to get similarity scores
    top_k_weights = normalize_similarity(top_k_distances)
    
    # Apply softmax normalization
    top_k_weights = torch.softmax(top_k_weights, dim=1)
    
    return top_k_indices, top_k_weights


def find_topk_similar_theta_batch(query_hashes, database_hashes, k, distance_metric='cosine'):
    """
    :param query_hashes: {-1,+1}^{M*q} batch of query hash codes
    :param database_hashes: {-1,+1}^{N*q} database hash codes
    :param k: number of top similar hashes to return
    :return: top_k_indices [M, k], top_k_weights [M, k]
    """
    # Calculate the Hamming distances between the query hashes and all database hashes
    if distance_metric == 'cosine':
        hamming_distances = calculate_cosine(query_hashes, database_hashes)
    elif distance_metric == 'euclidean':
        hamming_distances = calculate_euclidean(query_hashes, database_hashes)

    # Get the indices of the top k smallest Hamming distances
    top_k_indices = torch.topk(hamming_distances, k, largest=False, dim=1).indices

    # Get the top k distances
    top_k_distances = torch.gather(hamming_distances, 1, top_k_indices)

    # Normalize the distances to get similarity scores
    top_k_weights = normalize_similarity(top_k_distances)

    # Apply softmax normalization
    top_k_weights = torch.softmax(top_k_weights, dim=1)

    return top_k_indices, top_k_weights



# 根据不同距离改写find_topk_similar_hashes_batch
def find_topk_similar_hashes_batch_custom(query_hashes, database_hashes, k, distance_metric='hamming'):
    """
    :param query_hashes: {-1,+1}^{M*q} batch of query hash codes
    :param database_hashes: {-1,+1}^{N*q} database hash codes
    :param k: number of top similar hashes to return
    :param distance_metric: 'hamming', 'cosine', or 'euclidean'
    :return: top_k_indices [M, k], top_k_weights [M, k]
    """
    if distance_metric == 'hamming':
        distances = calculate_hamming(query_hashes, database_hashes)
    elif distance_metric == 'cosine':
        distances = calculate_cosine(query_hashes, database_hashes)
    elif distance_metric == 'euclidean':
        distances = calculate_euclidean(query_hashes, database_hashes)
    else:
        raise ValueError("Unsupported distance metric. Use 'hamming', 'cosine', or 'euclidean'.")

    # Get the indices of the top k smallest distances
    top_k_indices = torch.topk(distances, k, largest=False, dim=1).indices

    # Get the top k distances
    top_k_distances = torch.gather(distances, 1, top_k_indices)

    # Normalize the distances to get similarity scores
    top_k_weights = normalize_similarity(top_k_distances)

    # Apply softmax normalization
    top_k_weights = torch.softmax(top_k_weights, dim=1)

    return top_k_indices, top_k_weights