def invert_counts(counts):
    return {k[::-1]:v for k, v in counts.items()}
