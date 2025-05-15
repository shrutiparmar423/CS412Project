# Complete Wind Turbine MLP with FOAFs
import pandas as pd
import math
import random

# Load the dataset
df = pd.read_csv("wind_turbine.csv", sep=",")

# Create binary target (1 if power >= 1500, else 0)
df['label'] = (df['power'] >= 1500).astype(int)
features = df[['wind speed', 'wind direction']].values.tolist()
labels = df['label'].tolist()

# Normalize features
def normalize(data):
    cols = list(zip(*data))
    normalized = []
    for col in cols:
        min_val, max_val = min(col), max(col)
        # Avoid division by zero
        if min_val == max_val:
            normalized.append([0] * len(col))
        else:
            normalized.append([(x - min_val) / (max_val - min_val) for x in col])
    return list(map(list, zip(*normalized)))

X_norm = normalize(features)

# Train-test split
combined = list(zip(X_norm, labels))
random.shuffle(combined)
split_idx = int(len(combined) * 0.8)
train, test = combined[:split_idx], combined[split_idx:]
X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

# Activation functions
functions = {
    'sigmoid': (lambda x: 1 / (1 + math.exp(-x)),
                lambda x: (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x))))),
    'frac_sigmoid': (lambda x, alpha=0.5: (1 / (1 + math.exp(-x))) ** alpha,
                     lambda x, alpha=0.5: alpha * ((1 / (1 + math.exp(-x))) ** (alpha - 1)) * (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x))))),
    'tanh': (math.tanh, lambda x: 1 - math.tanh(x) ** 2),
    'frac_tanh': (lambda x, alpha=0.5: abs(math.tanh(x)) ** alpha * (1 if math.tanh(x) >= 0 else -1),
                  lambda x, alpha=0.5: alpha * abs(math.tanh(x)) ** (alpha - 1) * (1 - math.tanh(x) ** 2) if math.tanh(x) != 0 else 0),
    'relu': (lambda x: max(0, x), lambda x: 1 if x > 0 else 0),
    'frac_relu': (lambda x, alpha=0.5: x ** alpha if x > 0 else 0,
                  lambda x, alpha=0.5: alpha * (x ** (alpha - 1)) if x > 0 else 0)
}

# Simple MLP
class SimpleMLP:
    def __init__(self, input_size, hidden1, hidden2, act_type='sigmoid', alpha=1.0):
        self.alpha = alpha
        self.act_type = act_type
        self.act, self.act_deriv = functions[act_type]
        self.init_weights(input_size, hidden1, hidden2)

    def init_weights(self, input_size, h1, h2):
        self.w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(h1)]
        self.b1 = [0] * h1
        self.w2 = [[random.uniform(-1, 1) for _ in range(h1)] for _ in range(h2)]
        self.b2 = [0] * h2
        self.w3 = [random.uniform(-1, 1) for _ in range(h2)]
        self.b3 = 0

    def forward(self, x):
        self.z1 = [sum(x[i] * self.w1[j][i] for i in range(len(x))) + self.b1[j] for j in range(len(self.w1))]
        self.a1 = [self.act(z, self.alpha) if 'frac' in self.act_type else self.act(z) for z in self.z1]
        self.z2 = [sum(self.a1[i] * self.w2[j][i] for i in range(len(self.a1))) + self.b2[j] for j in range(len(self.w2))]
        self.a2 = [self.act(z, self.alpha) if 'frac' in self.act_type else self.act(z) for z in self.z2]
        self.z3 = sum(self.a2[i] * self.w3[i] for i in range(len(self.a2))) + self.b3
        self.a3 = self.act(self.z3, self.alpha) if 'frac' in self.act_type else self.act(self.z3)
        return self.a3

    def backward(self, x, y, lr=0.05):
        d3 = (self.a3 - y) * (self.act_deriv(self.z3, self.alpha) if 'frac' in self.act_type else self.act_deriv(self.z3))
        for i in range(len(self.w3)):
            self.w3[i] -= lr * d3 * self.a2[i]
        self.b3 -= lr * d3

        d2 = [d3 * self.w3[i] * (self.act_deriv(self.z2[i], self.alpha) if 'frac' in self.act_type else self.act_deriv(self.z2[i])) for i in range(len(self.z2))]
        for i in range(len(self.w2)):
            for j in range(len(self.w2[i])):
                self.w2[i][j] -= lr * d2[i] * self.a1[j]
            self.b2[i] -= lr * d2[i]

        d1 = [sum(d2[k] * self.w2[k][i] for k in range(len(self.w2))) * (self.act_deriv(self.z1[i], self.alpha) if 'frac' in self.act_type else self.act_deriv(self.z1[i])) for i in range(len(self.z1))]
        for i in range(len(self.w1)):
            for j in range(len(self.w1[i])):
                self.w1[i][j] -= lr * d1[i] * x[j]
            self.b1[i] -= lr * d1[i]

    def train(self, X, y, epochs=2000, lr=0.1):
        for _ in range(epochs):
            for i in range(len(X)):
                self.forward(X[i])
                self.backward(X[i], y[i], lr)

    def evaluate(self, X, y):
        correct = sum((self.forward(x) > 0.5) == y[i] for i, x in enumerate(X))
        return correct / len(X)

# Run all 6 activation types
configs = [
    ('sigmoid', 1.0), ('frac_sigmoid', 0.5),
    ('tanh', 1.0), ('frac_tanh', 0.5),
    ('relu', 1.0), ('frac_relu', 0.5)
]

print("Activation Type Comparison (Accuracy):")
for act, alpha in configs:
    model = SimpleMLP(2, 5, 3, act_type=act, alpha=alpha)
    model.train(X_train, y_train)
    acc = model.evaluate(X_test, y_test)
    print(f"Activation: {act:<12} α={alpha:<4} → Accuracy: {acc * 100:.2f}%")
