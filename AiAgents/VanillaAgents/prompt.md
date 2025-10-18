“Write a friendly, step-by-step spoken tutorial for non-experts that shows how to use Python classes to build a tiny agents system and workflow from scratch. Create one clear section per feature below. For each section:

Start with a short voiceover paragraph (the “script”) that explains the concept in plain language and why it matters for agents.

Include a concise, runnable code example (≤25 lines) that demonstrates the feature in a realistic mini-agent context.

Add a brief “Takeaway” line.

Use only the Python standard library; target Python 3.10+.

Prefer descriptive names and comments; include an example print/output comment when helpful.

Features to cover:

# Fundamentals (1–5)

1. **Minimal class & instance attributes**

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

p = Point(2, 3)
```

*Basic container for related data.*

2. **Default & keyword args in `__init__`**

```python
class User:
    def __init__(self, name, role="viewer", *, active=True):
        self.name, self.role, self.active = name, role, active

u = User("Ana", active=False)
```

*Clean, explicit initialization.*

3. **Class attributes vs instance attributes**

```python
class Counter:
    total = 0              # class attr (shared)
    def __init__(self):
        self.count = 0     # instance attr
        Counter.total += 1
```

*Know what’s shared vs per-object.*

4. **`__slots__` to save memory / prevent typos**

```python
class Row:
    __slots__ = ("id", "name")
    def __init__(self, id, name):
        self.id, self.name = id, name
```

*Avoids per-instance `__dict__`; blocks unknown attrs.*

5. **`dataclasses` for boilerplate-free models**

```python
from dataclasses import dataclass

@dataclass
class Product:
    sku: str
    price: float
    in_stock: bool = True

p = Product("A1", 9.99)
```

*Auto `__init__`, `__repr__`, comparisons, etc.*

---