def candidate_elimination(examples):
    # Initialize the specific and general hypotheses
    S = [None] * len(examples[0])  # most specific hypothesis
    G = [['?'] * len(examples[0])]  # most general hypothesis

    # Check each training example
    for example in examples:
        x, y = example[:-1], example[-1]  # Input features and target label

        if y == 'Y':  # Positive example
            # Remove inconsistent hypotheses from G
            G = [g for g in G if consistent(g, x)]

            # Generalize S with x
            for i in range(len(S)):
                if S[i] != x[i]:
                    S[i] = '?' if S[i] != '?' else x[i]

            # Remove hypotheses from G that are more general than any other hypothesis in G
            G = [g for g in G if not any(more_general(h, g) for h in G)]

        else:  # Negative example
            # Remove inconsistent hypotheses from S
            S = [s for s in S if not consistent(s, x)]

            # Specialize G with x
            G_prime = []
            for g in G:
                if consistent(g, x):
                    for i in range(len(g)):
                        if g[i] == '?':
                            g_prime = list(g)
                            g_prime[i] = x[i]
                            if any(more_general(h, g_prime) for h in G):
                                continue
                            G_prime.append(g_prime)
                else:
                    G_prime.append(g)
            G = G_prime

    return S, G


def consistent(hypothesis, example):
    for h, x in zip(hypothesis, example):
        if h != '?' and h != x:
            return False
    return True

# Example training data
training_examples = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Y'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Y'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'N'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Y']
]

S, G = candidate_elimination(training_examples)

print("Specific hypotheses (S):")
for s in S:
    print(s)

print("\nGeneral hypotheses (G):")
for g in G:
    print(g)

def more_general(h1, h2):
    for x, y in zip(h1, h2):
        if x != '?' and (x != y or y == '?'):
            return False
    return True
