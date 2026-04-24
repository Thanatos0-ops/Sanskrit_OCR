# Step 1: Edit distance (core of both metrics)

def edit_distance(s1, s2):
    """Computes levenshtein distance between two strings
    """

    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        dp[i][0] = i

    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):

            if s1[i - 1] == s2[j -1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,           # deletion
                dp[i][j - 1] + 1,           # insertion
                dp[i - 1][j - 1] + cost     # substitution
            )

    return dp[-1][-1]


# Step 2: CER (Character Error Rate)

def cer(pred, target):
    """
    Character Error Rate
    """

    if len(target) == 0:
        return 0.0 
    
    return edit_distance(pred, target) / len(target)


# Step 3: WER (Word Error Rate)

def wer(pred, target):
    """
    Word Error Rate
    """

    pred_words = pred.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 0.0
    
    return edit_distance(pred_words, target_words) / len(target_words)