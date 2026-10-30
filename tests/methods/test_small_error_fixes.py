import pytest

from gradeflow_core.methods.code.fixes import FixedCode, fix_small_errors, fixed_code


def fixed(code: str, defined: tuple[str, ...] = (), limit: int = 5) -> FixedCode | None:
    return fix_small_errors(code, defined, limit)


@pytest.mark.parametrize(
    ("code", "expected", "fix"),
    [
        (
            "def add(a, b)\n    return a + b\n",
            "def add(a, b):\n    return a + b\n",
            "line 1, added “:” after “def add(a, b)”",
        ),
        (
            "if x > 0:\n    y = 1\nelse\n    y = 2\n",
            "if x > 0:\n    y = 1\nelse:\n    y = 2\n",
            "line 3, added “:” after “else”",
        ),
        (
            "for i in range(3)  # count\n    print(i)\n",
            "for i in range(3):  # count\n    print(i)\n",
            "line 1, added “:” after “for i in range(3)”",
        ),
        ('print("hi"\nx = 1\n', 'print("hi")\nx = 1\n', "line 1, closed “(”"),
        ("total = sum([1, 2\n", "total = sum([1, 2])\n", "line 1, closed “([”"),
        (
            "if (x > 0:\n    y = 1\n",
            "if (x > 0):\n    y = 1\n",
            "line 1, closed “(”",
        ),
        ("print(\n    a,\n    b\nx = 1\n", "print(\n    a,\n    b)\nx = 1\n", "line 3, closed “(”"),
        ('print("hi)\n', 'print("hi")\n', 'line 1, closed “"”'),
        ("s = 'abc\n", "s = 'abc'\n", "line 1, closed “'”"),
        (
            'if answer == "yes:\n    y = 1\n',
            'if answer == "yes":\n    y = 1\n',
            'line 1, closed “"”',
        ),
        ("if x = 1:\n    y = 1\n", "if x == 1:\n    y = 1\n", "line 1, “=” read as “==”"),
        (
            "x = 3\nwhile x = 3:\n    x = 4\n",
            "x = 3\nwhile x == 3:\n    x = 4\n",
            "line 2, “=” read as “==”",
        ),
        (
            "def f(x):\n    retrun x\n",
            "def f(x):\n    return x\n",
            "line 2, “retrun” read as “return”",
        ),
        (
            "x = 1\nwhlie x < 3:\n    x += 1\n",
            "x = 1\nwhile x < 3:\n    x += 1\n",
            "line 2, “whlie” read as “while”",
        ),
        ("Def f(x):\n    return x\n", "def f(x):\n    return x\n", "line 1, “Def” read as “def”"),
        (
            "def f(x):\n    Return x\n",
            "def f(x):\n    return x\n",
            "line 2, “Return” read as “return”",
        ),
        (
            "def f(x):\n    Return(x)\n",
            "def f(x):\n    return(x)\n",
            "line 2, “Return” read as “return”",
        ),
        ("for i in []:\n    brek\n", "for i in []:\n    break\n", "line 2, “brek” read as “break”"),
        ("x = true\n", "x = True\n", "line 1, “true” read as “True”"),
        ("x = [false, none]\n", "x = [False, None]\n", None),
        ("x = Flase\n", "x = False\n", "line 1, “Flase” read as “False”"),
        (
            "length = 3\nprint(lenght)\nprint(lenght + 1)\n",
            "length = 3\nprint(length)\nprint(length + 1)\n",
            "line 2, “lenght” read as “length”",
        ),
        ('Print("hi")\n', 'print("hi")\n', "line 1, “Print” read as “print”"),
    ],
)
def test_fixes_each_small_error(code: str, expected: str, fix: str | None) -> None:
    result = fixed(code)
    assert result is not None
    assert result.code == expected
    if fix is not None:
        assert result.fixes == (fix,)


def test_fixes_several_errors_within_the_limit() -> None:
    code = "def add(a, b)\n    retrun a + b\nprint(add(1, 2)\n"
    assert fixed(code, limit=3) == FixedCode(
        "def add(a, b):\n    return a + b\nprint(add(1, 2))\n",
        (
            "line 1, added “:” after “def add(a, b)”",
            "line 2, “retrun” read as “return”",
            "line 3, closed “(”",
        ),
    )


def test_gives_nothing_over_the_limit() -> None:
    code = "def add(a, b)\n    retrun a + b\nprint(add(1, 2)\n"
    assert fixed(code, limit=2) is None
    assert fixed("x = true\n", limit=0) is None


@pytest.mark.parametrize(
    "code",
    [
        "def add(a, b):\n    return a + b\n",
        "print(undefined_thing)\n",
        "",
    ],
)
def test_gives_nothing_without_fixes(code: str) -> None:
    assert fixed(code) is None


@pytest.mark.parametrize(
    "code",
    [
        "def f(x)\n    return x +\n",
        "if x > 0 return 1\n",
        "x = (1, 2]\n",
        's = """abc\n',
        "def f(x)\nreturn x\n    y = 1\n",
    ],
)
def test_gives_nothing_when_still_broken(code: str) -> None:
    assert fixed(code) is None


def test_leaves_strings_and_comments_alone() -> None:
    code = 'length = 1\ns = "if x = 1 retrun lenght true"  # whlie lenght true\nprint(lenght, s)\n'
    assert fixed(code) == FixedCode(
        'length = 1\ns = "if x = 1 retrun lenght true"  # whlie lenght true\nprint(length, s)\n',
        ("line 3, “lenght” read as “length”",),
    )


@pytest.mark.parametrize(
    "condition",
    ["x == 1", "x != 1", "x <= 1", "x >= 1", "(y := x)", "f(key=1)", 'f"{x=}"'],
)
def test_leaves_comparisons_walrus_and_keyword_arguments(condition: str) -> None:
    code = f"def f(key=0):\n    return key\nx = 1\nif {condition}\n    pass\n"
    result = fixed(code)
    assert result is not None
    assert result.fixes == (f"line 4, added “:” after “if {condition}”",)


def test_leaves_a_name_close_to_two_names() -> None:
    assert fixed("cat = 1\ncar = 2\nprint(cap)\n") is None


def test_leaves_short_names() -> None:
    assert fixed("i = 1\nprint(n)\n") is None


def test_counts_names_the_setup_defines() -> None:
    assert fixed("print(answer)\n", defined=("answer",)) is None
    result = fixed("print(anwser)\n", defined=("answer",))
    assert result is not None
    assert result.code == "print(answer)\n"


def test_counts_names_the_setup_binds_but_not_those_it_only_reads() -> None:
    setup = "try:\n    pass\nexcept ValueError as error:\n    pass\nprint(data)"
    assert fixed_code("print(dta)\n", setup, 1) is None
    result = fixed_code("print(eror)\n", setup, 1)
    assert result is not None
    assert result.code == "print(error)\n"


def test_leaves_a_true_the_student_defines() -> None:
    assert fixed("true = 1\nprint(true)\n") is None


def test_leaves_an_ambiguous_keyword() -> None:
    assert fixed("fro i in range(3):\n    print(i)\n") is None


def test_fixes_flat_code_then_lays_it_out() -> None:
    code = "def total(items)\nresult = 0\nfor item in items:\nresult += item\nretrun result\n"
    result = fixed(code)
    assert result is not None
    assert result.fixes == (
        "line 1, added “:” after “def total(items)”",
        "line 5, “retrun” read as “return”",
    )
    compile(result.code, "<code>", "exec")
    assert result.code.splitlines()[0] == "def total(items):"
    assert result.code.splitlines()[1].startswith(" ")


def test_fixes_names_in_flat_code_after_the_layout() -> None:
    code = "def f(length):\nreturn lenght\n"
    result = fixed(code)
    assert result is not None
    assert result.fixes == ("line 2, “lenght” read as “length”",)
    assert result.code.splitlines()[1].strip() == "return length"
