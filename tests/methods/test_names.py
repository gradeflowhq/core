import pytest

from gradeflow_core import BUILTINS, MethodFileError
from gradeflow_core.editing import method_name, method_ref, split_ref


def test_a_saved_method_is_named_in_its_library() -> None:
    assert (
        method_name("intro-biology", "term-not-look-alike") == "intro-biology/term-not-look-alike"
    )
    assert method_ref("intro-biology/term", 3) == "intro-biology/term@3"


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("Ana", "term"),
        ("ana", "term_1"),
        ("ana", "-term"),
        ("an--a", "term"),
        ("", "term"),
        ("a" * 61, "x"),
        ("ana", "x" * 101),
    ],
)
def test_namespaces_and_names_are_lowercase_words_joined_by_hyphens(
    namespace: str, name: str
) -> None:
    with pytest.raises(MethodFileError):
        method_name(namespace, name)


def test_namespaces_and_names_have_their_own_limits() -> None:
    assert method_name("a" * 60, "x" * 100)


def test_the_core_keeps_its_namespace() -> None:
    with pytest.raises(MethodFileError, match="GradeFlow's own"):
        method_name("gradeflow", "key-points")


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("gradeflow/term", "gradeflow/term: gradeflow is GradeFlow's own."),
        ("term", "term: Name it as a namespace and name"),
        ("Ana/term", "Ana/term: Use lowercase letters"),
    ],
)
def test_loaded_methods_are_named_as_saved_methods_are(name: str, message: str) -> None:
    text = "version: 1\nlabel: Term\nsummary: s\nquestion_types: [text]\nchecks: []\n"
    with pytest.raises(MethodFileError, match=message):
        BUILTINS.with_files([(name, text)])


def test_a_reference_splits_into_its_parts() -> None:
    assert split_ref("ana/term-not-look-alike@12") == ("ana", "term-not-look-alike", 12)
    assert all(split_ref(method.ref)[0] == "gradeflow" for method in BUILTINS)


@pytest.mark.parametrize("ref", ["ana/term", "term@1", "ana/term@0", "ana/term@v1", "a/b/c@1"])
def test_a_reference_names_a_library_method_and_version(ref: str) -> None:
    with pytest.raises(MethodFileError):
        split_ref(ref)
