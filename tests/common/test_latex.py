import pytest

from gradeflow_core.common.latex import latex_text


@pytest.mark.parametrize(
    ("latex", "text"),
    [
        (r"s_{0}=\text{ABC}", "s_0=ABC"),
        ("y^{(i)}", "y^(i)"),
        ("x_{i+1}", "x_(i+1)"),
        ("h^{*}(s_{0})", "h^*(s_0)"),
        (r"\frac{1}{2n}", "1/(2n)"),
        (r"\operatorname{max}(a) + \max b + \operatorname{log}n", "max(a)+max b+log n"),
        (r"\sqrt{x}\ \sqrt[3]{y}", "sqrt(x) root(y, 3)"),
        (r"\begin{matrix}1 & 2 \\ 3\end{matrix}", "1, 2; 3"),
        (r"A=\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}", "A=[1, 2; 3, 4]"),
        (r"x \times y \leq -1 \ne \alpha", "x×y<=-1!=alpha"),
        (r"\left( \frac{a+b}{c} \right)", "((a+b)/c)"),
        (r"\text{if } x > 0", "if x>0"),
        (r"\{a\} \$", "{a}$"),
        (r"a \backslash b\text{c\textasciicircum d\textasciitilde}", "a\\bc^d~"),
        ("x^", "x^"),
        (r"\frac{1}{", "1/"),
    ],
)
def test_latex_reads_as_people_type_it(latex: str, text: str) -> None:
    assert latex_text(latex) == text


def test_latex_nested_too_deeply_stays_as_it_is() -> None:
    latex = "{" * 5000 + "x" + "}" * 5000
    assert latex_text(latex) == latex
