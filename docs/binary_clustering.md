# Clustering Binary Survey Responses


#### Imports

```python
import numpy as np
from noloox import DirichletMultinomialMixture

```

#### Simulating data

```python
generator = np.random.default_rng(42)

columns = ["engineers", "academics", "doctors", "lawyers", "artists", "musicians", "skilled worker", "other"]
true_prob = [
    [0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05], # Academic orientation
    [0.4, 0, 0.05, 0.05, 0., 0.05, 0.3, 0.15], # Technical orientation
    [0, 0.1, 0.05, 0.05, 0.3, 0.4, 0, 0.1], # Art orientation
]
weights = [0.2, 0.7, 0.1]

n_schools = 200
n_students_per_school = 30
X = []
for i in n_schools:
    school_type = generator.choice(np.arange(3))
```
