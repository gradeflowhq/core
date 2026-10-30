"""Guessing the indentation exam software dropped from Python code."""

import time

import pytest

from gradeflow_core.questions.layouts import guess_layouts

TUNED = {
    "first_positive": """def first_positive(xs):
    for x in xs:
        if x > 0:
            return x
    return None""",
    "count_words": """def count_words(text):
    counts = {}
    for word in text.split():
        word = word.lower()
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    best = None
    for word, n in counts.items():
        if best is None or n > counts[best]:
            best = word
    return counts, best""",
    "total_positive": """def total_positive(xs):
    total = 0
    for x in xs:
        if x > 0:
            total += x
    return total""",
    "transpose": """def transpose(m):
    result = []
    for j in range(len(m[0])):
        row = []
        for i in range(len(m)):
            row.append(m[i][j])
        result.append(row)
    return result""",
    "gcd": """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a""",
    "fact": """def fact(n):
    if n == 0:
        return 1
    return n * fact(n - 1)""",
    "search": """def search(xs, t):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == t:
            return mid
        elif xs[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1""",
    "two_functions": """def square(x):
    return x * x
def sum_squares(xs):
    return sum(square(x) for x in xs)""",
    "stack": """class Stack:
    def __init__(self):
        self.items = []
    def push(self, x):
        self.items.append(x)
    def pop(self):
        return self.items.pop()""",
    "parse": """def parse(s):
    try:
        n = int(s)
    except ValueError:
        n = 0
    return n""",
    "evens": """def evens(xs):
    out = []
    for x in xs:
        if x % 2 == 0:
            out.append(x)
    print(len(out))
    return out""",
    "bubble": """def bubble(xs):
    n = len(xs)
    for i in range(n):
        for j in range(n - i - 1):
            if xs[j] > xs[j + 1]:
                xs[j], xs[j + 1] = xs[j + 1], xs[j]
    return xs""",
    "find": """def find(xs, t):
    found = False
    for x in xs:
        if x == t:
            found = True
            break
    return found""",
    "stats": """def stats(xs):
    lo = hi = xs[0]
    for x in xs:
        if x < lo:
            lo = x
        if x > hi:
            hi = x
    return lo, hi""",
    "bfs": """def bfs(graph, start):
    seen = {start}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen""",
}

# Never used to choose or tune a rule.
HELD_OUT = {
    "grade": """def grade(score):
    if score >= 80:
        letter = "A"
    elif score >= 60:
        letter = "B"
    else:
        letter = "C"
    return letter""",
    "digits": """def digit_sum(n):
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total""",
    "reverse_words": """def reverse_words(s):
    words = s.split()
    out = ""
    for w in words:
        out = w + " " + out
    return out.strip()""",
    "group_by": """def group_by_length(words):
    groups = {}
    for w in words:
        k = len(w)
        if k not in groups:
            groups[k] = []
        groups[k].append(w)
    return groups""",
    "skip_negatives": """def mean_positive(xs):
    total = 0
    count = 0
    for x in xs:
        if x <= 0:
            continue
        total += x
        count += 1
    if count == 0:
        return 0
    return total / count""",
    "safe_div": """def safe_ratios(pairs):
    out = []
    for a, b in pairs:
        try:
            out.append(a / b)
        except ZeroDivisionError:
            out.append(None)
    return out""",
    "matrix_sum": """def matrix_sum(m):
    total = 0
    for row in m:
        for v in row:
            total += v
    return total""",
    "two_loops": """def common(xs, ys):
    seen = set()
    for x in xs:
        seen.add(x)
    out = []
    for y in ys:
        if y in seen:
            out.append(y)
    return out""",
    "is_prime": """def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True""",
    "helper": """def mean(xs):
    return sum(xs) / len(xs)
def variance(xs):
    m = mean(xs)
    total = 0
    for x in xs:
        total += (x - m) ** 2
    return total / len(xs)""",
    "fizzbuzz": """def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out""",
    "max_run": """def longest_run(xs):
    best = 0
    run = 0
    prev = None
    for x in xs:
        if x == prev:
            run += 1
        else:
            run = 1
        best = max(best, run)
        prev = x
    return best""",
    "count_vowels": """def count_vowels(s):
    n = 0
    for ch in s.lower():
        if ch in "aeiou":
            n += 1
    return n""",
    "prefix_sums": """def prefix_sums(xs):
    out = []
    total = 0
    for x in xs:
        total += x
        out.append(total)
    return out""",
    "flatten": """def flatten(nested):
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat""",
    "account": """class Account:
    def __init__(self, balance):
        self.balance = balance
    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        return self.balance""",
    "index_of": """def index_of(xs, t):
    i = 0
    while i < len(xs):
        if xs[i] == t:
            return i
        i += 1
    return -1""",
    "most_common": """def most_common(words):
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    best = max(counts, key=counts.get)
    return best""",
    "pairs": """def pairs_summing_to(xs, target):
    found = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == target:
                found.append((i, j))
    return found""",
    "read_numbers": """def read_numbers(lines):
    numbers = []
    for line in lines:
        try:
            numbers.append(float(line))
        except ValueError:
            print("skipping", line)
    return numbers""",
    "collatz": """def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps""",
    "is_palindrome": """def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True""",
}


def flat(code: str) -> str:
    return "\n".join(line.lstrip() for line in code.splitlines())


def rank(code: str) -> int | None:
    """Where the layout meant comes among the first three guesses, from 1."""
    guesses = guess_layouts(flat(code), 3)
    return guesses.index(code) + 1 if code in guesses else None


@pytest.mark.parametrize("name", TUNED)
def test_tuned_cases_come_first(name: str) -> None:
    assert rank(TUNED[name]) == (2 if name == "bfs" else 1)


@pytest.mark.parametrize("name", HELD_OUT)
def test_held_out_cases_come_in_the_first_three(name: str) -> None:
    assert rank(HELD_OUT[name]) is not None


@pytest.mark.parametrize(
    "code",
    ["x = 1\nprint(x)", "def f(x):\n  if x:\n    return 1\n  return 2", ""],
)
def test_code_compiling_as_typed_is_its_only_layout(code: str) -> None:
    assert guess_layouts(code, 3) == [code]


@pytest.mark.parametrize(
    "code", ["def f(:\nreturn 1", "return 1", "x = (1,\n2", "for x in xs:\nprint(x))"]
)
def test_code_no_layout_compiles_has_none(code: str) -> None:
    assert guess_layouts(code, 3) == []


def test_leftover_indentation_is_ignored() -> None:
    code = TUNED["first_positive"]
    assert guess_layouts(code.replace("\n        ", "\n "), 1) == [code]


def test_limit_caps_the_layouts() -> None:
    code = flat("def f(xs):\nfor x in xs:\nif x:\nprint(x)\nprint(xs)")
    guesses = guess_layouts(code, 10)
    assert len(guesses) > 2
    assert guess_layouts(code, 2) == guesses[:2]


def test_brackets_continue_their_statement() -> None:
    code = """def f(xs):
    total = sum(
        x
        for x in xs
    )
    return {
        "total": total,
    }"""
    assert guess_layouts(flat(code), 1) == [code]


def test_backslash_continues_its_statement() -> None:
    code = "def f(a, b):\n    total = a + \\\n        b\n    return total"
    assert guess_layouts(flat(code), 1) == [code]


def test_string_lines_stay_as_typed() -> None:
    code = '''def f(x):
    """Doc.

  indented inside
"""
    s = f"""{x}
 # not a comment
"""
    return s'''
    typed = code.replace("\n    ", "\n")
    assert guess_layouts(typed, 1) == [code]


def test_strings_and_comparisons_do_not_read_as_names_or_assignments() -> None:
    code = """def f(xs):
    out = []
    for x in xs:
        if x == "out = x":
            out.append(x)
    print("x")
    return out"""
    assert guess_layouts(flat(code), 1) == [code]


def test_comments_take_the_depth_of_the_statement_after() -> None:
    code = """# totals
def f(xs):
    # start at zero
    total = 0
    for x in xs:
        # add it
        total += x  # running
    return total
    # done"""
    assert guess_layouts(flat(code), 1) == [code]


def test_blank_lines_stay_blank() -> None:
    code = "def a():\n    return 1\n\n\ndef b():\n\n    return 2"
    assert guess_layouts(flat(code), 1) == [code]


def test_continuation_keywords_line_up_with_their_block() -> None:
    code = """def f(path):
    try:
        data = open(path).read()
    except OSError:
        return None
    else:
        if data:
            return data
        elif data is None:
            raise ValueError(path)
        else:
            return ""
    finally:
        print(path)"""
    assert guess_layouts(flat(code), 1) == [code]


def test_one_line_blocks_can_be_continued() -> None:
    code = "def f(x):\n    if x: y = 1\n    else: y = 2\n    return y"
    assert guess_layouts(flat(code), 1) == [code]


def test_decorated_methods_stay_in_their_class() -> None:
    code = """class Point:
    def __init__(self, x):
        self.x = x
    @property
    def double(self):
        return self.x * 2
    @staticmethod
    def origin():
        return Point(0)"""
    assert guess_layouts(flat(code), 1) == [code]


def test_lines_using_a_functions_names_stay_in_it() -> None:
    code = """def greet(name):
    message = "Hi " + name
    print(message)
print(greet("Ana"))"""
    assert guess_layouts(flat(code), 1) == [code]


def test_long_answers_are_fast() -> None:
    body = flat(TUNED["count_words"])
    code = "\n".join(body.replace("count_words", f"count_{i}") for i in range(8))
    start = time.perf_counter()
    guesses = guess_layouts(code, 3)
    assert time.perf_counter() - start < 1
    assert len(code.splitlines()) > 100
    assert guesses


def test_long_functions_are_fast() -> None:
    loops = [f"for x{i} in xs:\nif x{i}:\ntotal += x{i}" for i in range(33)]
    code = "\n".join(["def f(xs):", "total = 0", *loops, "return total"])
    start = time.perf_counter()
    guesses = guess_layouts(code, 3)
    assert time.perf_counter() - start < 1
    assert guesses


def test_hopeless_long_answers_stop_in_time() -> None:
    code = "\n".join(["x = 0", *["if x:\nx += 1"] * 50, "return x"])
    start = time.perf_counter()
    assert guess_layouts(code, 3) == []
    assert time.perf_counter() - start < 1
