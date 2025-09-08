#!/usr/bin/env python3
# Test the code labeler with proper markdown format
from labeling.code_labeler import CodeLabeler

labeler = CodeLabeler()

sample_code = '''```python
def calculate_metrics(data, threshold=0.5):
    """Calculate accuracy metrics."""
    from sklearn.metrics import accuracy_score
    return accuracy_score(data, threshold)
```'''

print('Testing code labeler with markdown format:')

symbols = labeler.extract_code_symbols(sample_code, 'python')
print(f'Symbols extracted: {len(symbols)}')
for symbol in symbols:
    print(f'  - {symbol}')

# Also test without language hint
print('\nTesting without language hint:')
symbols2 = labeler.extract_code_symbols(sample_code)
print(f'Symbols extracted: {len(symbols2)}')