import pytest

from gradeflow_core.adapters.mathml import mathml_latex


@pytest.mark.parametrize(
    ("body", "latex"),
    [
        ("<msub><mi>s</mi><mn>0</mn></msub><mo>=</mo><mtext>ABC</mtext>", r"s_{0}=\text{ABC}"),
        ("<msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup>", "y^{(i)}"),
        ("<msub><mi>x</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub>", "x_{i+1}"),
        ("<mfrac><mn>1</mn><mrow><mn>2</mn><mi>n</mi></mrow></mfrac>", r"\frac{1}{2n}"),
        ("<mi>max</mi><mo>⁡</mo><mo>(</mo><mi>a</mi><mo>)</mo>", r"\operatorname{max}(a)"),
        (
            "<msqrt><mi>x</mi></msqrt><mspace/><mroot><mi>y</mi><mn>3</mn></mroot>",
            r"\sqrt{x}\ \sqrt[3]{y}",
        ),
        (
            "<mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd></mtr></mtable>",
            r"\begin{matrix}1 & 2 \\ 3\end{matrix}",
        ),
        ("<mi>x</mi><mo>×</mo><mi>y</mi><mo>≤</mo><mo>\u2212</mo><mn>1</mn>", r"x\times y\le -1"),
        ("<mo>{</mo><mi>a</mi><mo>}</mo><mtext>50% of $5</mtext>", r"\{a\}\text{50\% of \$5}"),
        (
            "<mi>A</mi><mo>=</mo><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo>"
            "<mo>(</mo><mtable><mtr><mtd><mn>2</mn></mtd></mtr></mtable><mo>]</mo>",
            r"A=\begin{bmatrix}1\end{bmatrix}(\begin{matrix}2\end{matrix}]",
        ),
        (
            "<msubsup><mi>s</mi><mn>0</mn><mo>\u2032</mo></msubsup><mi>w</mi><mo>\u2032</mo>",
            "s_{0}'w'",
        ),
        ("<msup><msup><mi>x</mi><mn>2</mn></msup><mn>3</mn></msup>", "{x^{2}}^{3}"),
        ("<munderover><mo>∑</mo><mi>i</mi><mi>n</mi></munderover><mi>i</mi>", r"\sum_{i}^{n}i"),
        ("<mspace/><mi>x</mi><mspace/>", "x"),
        (
            "<mi>a</mi><mo>^</mo><mo>~</mo><mo>\\</mo><mi>b</mi><mtext>c^d~e\\f</mtext>",
            r"a\text{\textasciicircum}\sim \backslash b"
            r"\text{c\textasciicircum d\textasciitilde e\textbackslash f}",
        ),
    ],
)
def test_math_reads_as_latex(body: str, latex: str) -> None:
    assert mathml_latex(f'<math xmlns="http://www.w3.org/1998/Math/MathML">{body}</math>') == latex
