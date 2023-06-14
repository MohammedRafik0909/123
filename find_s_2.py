def find_s(examples):
    hypothesis = examples[0].copy()  # Initialize the hypothesis with the first example

    for example in examples:
        if example['classification'] == 'Positive':
            for attr in example.keys():
                if example[attr] != hypothesis[attr]:
                    hypothesis[attr] = '?'  # Generalize the attribute

    return hypothesis


examples = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'classification': 'Negative'},
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'classification': 'Negative'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'classification': 'Positive'},
    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'classification': 'Positive'},
]

hypothesis = find_s(examples)
print(hypothesis)














                            
