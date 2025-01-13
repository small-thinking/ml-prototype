# This file implements the calculation of the AUC.
def calculate_auc(predicted_scores: list, labels: list) -> float:
    """
    Calculate the AUC (Area Under the Receiver Operating Characteristic Curve) of a model
    given the predicted scores and the true labels using an efficient and accurate method.

    :param predicted_scores: A list of predicted scores.
    :param labels: A list of true binary labels (0 or 1).
    :return: The AUC of the model.
    """

    # Step 1: Sort the list in ascending order based on predicted scores
    score_label_pairs = list(sorted(zip(predicted_scores, labels), key=lambda x: x[0]))

    # Step 2: Calculate the sum of ranks for positive samples
    sum_positive_ranks = 0.0
    num_positive = 0
    num_negative = 0
    for rank, (_, label) in enumerate(score_label_pairs):
        if label == 1:
            sum_positive_ranks += rank + 1
            num_positive += 1
        else:
            num_negative += 1

    # Step 3: Handle the case where there are no positive or no negative samples
    if num_positive == 0 or num_negative == 0:
        raise ValueError("The data must contain at least one positive and one negative sample.")

    # Step 4: Calculate the AUC using the Mann-Whitney U statistic
    # Calculate the AUC using the formula for the Mann-Whitney U statistic
    u_statistic = sum_positive_ranks - num_positive * (num_positive + 1) / 2
    auc = u_statistic / (num_positive * num_negative)
    return auc


if __name__ == '__main__':
    # Test the AUC calculation.
    predicted_scores = [0.1, 0.4, 0.35, 0.8, 0.1, 0.1, 0.1, 0.7]
    labels = [0, 1, 1, 0, 0, 0, 0, 1]
    print(calculate_auc(predicted_scores, labels))
