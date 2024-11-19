import json
from collections import defaultdict

import click
from rich.console import Console
from rich.table import Table

from typing import TypedDict


class Error(TypedDict):
    severity: str
    code: str


# Define the error categories
CATEGORIES: dict[str, str] = {
    "attr-defined": "Attribute Access: Attempting to access an attribute that doesn't exist",
    "arg-type": "Argument Type: Function argument has an incompatible type",
    "call-arg": "Call Argument: Invalid arguments passed to a function or method",
    "call-overload": "Call Overload: No overload matches the arguments passed",
    "assignment": "Assignment: Incompatible types in assignment",
    "index": "Indexing: Indexing operation on an invalid type",
    "name-defined": "Name Error: Name is not defined or used before assignment",
    "method-assign": "Method Assignment: Cannot assign to a method",
    "overload-cannot-match": "Overload Cannot Match: function signature will never be matched",
    "operator": "Operator: Unsupported operation for the types involved",
    "return-value": "Return Value: Return type is incompatible with the declared type",
    "union-attr": "Union Attribute: Attribute access on a union type not available in all members",
    "unreachable": "Unreachable Code: Code detected that is never executed",
    "misc": "Miscellaneous: General errors that don't fall into other categories",
    "attr-assign": "Attribute Assignment: Assignment of an attribute with an incompatible type",
    "assignment-none": "Assignment None: Assigning None to a non-optional type",
    "attr-assignment": "Attribute Assignment: Invalid assignment to an attribute",
    "redundant-cast": "Redundant Cast: Casting to the same type or redundant type casting",
    "redundant-expr": "Redundant Expression: Expression doesn't affect the outcome",
    "redundant-isinstance": "Redundant isinstance: isinstance check that is always true or false",
    "redundant-typevar": "Redundant TypeVar: Type variable usage is redundant",
    "import-untyped": "Import Untyped: Imported module is not typed",
    "callable-arg": "Callable Argument: Incorrect number or type of arguments passed to a callable",
    "var-annotated": "Variable Annotations: Variable has a wrong or missing type annotation",
    "var-missing": "Variable Missing: A variable is missing or undefined",
    "truthy-bool": "Boolean Context: Object in a boolean context without a clear truth value",
    "type-arg": "Type Argument: Invalid type argument provided",
    "type-var": "Type Variable: Misuse of a type variable",
    "unsubscriptable": "Unsubscriptable: Attempt to subscript a non-subscriptable type",
    "check-untyped-def": "Untyped Function: Function has no type annotations",
    "valid-type": "Type Validity: Invalid type annotation or comment",
    "var-no-assigned": "Variable Not Assigned: Variable declared but not assigned a value",
    "var-incompatible": "Variable Incompatible: Variable assigned with an incompatible type",
    "var-no-annotated": "Variable No Annotated: Missing type annotation where required",
    "var-implicit-none": "Implicit None: Variable is implicitly None potentially leading to errors",
    "redundant-operation": "Redundant Operation: Unnecessary operations like adding zero",
    "return-incompatible": "Return Incompatible: Return value doesn't match the function's return type",
    "return-not-required": "Return Not Required: Return statement in a function that doesn't need one",
    "shadowed": "Shadowed Name: A name is shadowed by another name in the same scope",
    "star-import": "Star Import: Star imports (e.g. `from module import *`) are discouraged",
    "str-bytes-concat": "String Bytes Concatenation: Concatenating a string with bytes is not allowed",
    "syntax": "Syntax: General syntax errors detected",
    "too-many-locals": "Too Many Locals: Function has too many local variables",
    "too-many-returns": "Too Many Returns: Function has too many return statements",
    "too-many-statements": "Too Many Statements: Function has too many statements; consider refactoring",
    "tuple-index": "Tuple Index: Invalid index operation on a tuple",
    "tuple-item": "Tuple Item: Invalid tuple item access",
    "used-before-definition": "Used Before Definition: Variable used before it was defined",
    "unused-ignore": "Unused Ignore: `# type: ignore` comment is unnecessary",
    "untyped-attr": "Untyped Attribute: Attribute access on an untyped object",
    "untyped-def": "Untyped Function Definition: Function or method lacks type annotations",
    "untyped-exception": "Untyped Exception: Raising or handling an exception without typing",
    "untyped-call": "Untyped Call: Calling a function or method without type annotations",
    "var-shadowed": "Variable Shadowed: Variable is shadowed by another variable in the same scope",
    "yield-from": "Yield From: Incorrect use of the `yield from` statement",
    "yield-expected": "Yield Expected: Missing yield statement in a generator function",
    "literal-required": "Literal Required: Non-literal type provided where a literal is required",
    "literal-comparison": "Literal Comparison: Unsafe or incompatible literal comparison",
    "super-init": "Super Init: Issues with calling the superclass `__init__` method",
    "redundant-else": "Redundant Else: Else clause is unnecessary because conditions are exhaustive",
    "redundant-typevar-bound": "Redundant TypeVar Bound: TypeVar bounds are redundant or unnecessary",
    "redundant-optional": "Redundant Optional: Optional type is redundant",
    "redundant-f-string": "Redundant f-string: f-string usage where a regular string would suffice",
    "simplifiable-if": "Simplifiable If: If statement can be simplified",
    "simplifiable-union": "Simplifiable Union: Union type can be simplified",
    "missing-type-var": "Missing TypeVar: TypeVar missing in function or class",
    "overload-impl": "Overload Implementation: Implementation doesn't match overload signatures",
    "var-hint-comment": "Variable Hint Comment: Issues with type hints provided as comments",
    "no-redef": "Redefinition: A name is redefined in an incompatible way",
    "str-format": "String Format: Issues with string formatting or interpolation",
    "yield-value": "Yield Value: Issues with values yielded by a generator function",
    "literal-bounds": "Literal Bounds: Literal value out of bounds",
    "dict-arg": "Dictionary Argument: Invalid arguments passed to a dictionary operation",
    "dict-item": "Dictionary Item: Access or assignment of a dictionary item with an incompatible type",
    "list-item": "List Item: Access or assignment of a list item with an incompatible type",
    "set-item": "Set Item: Access or assignment of a set item with an incompatible type",
    "none-return": "None Return: Returning `None` where a non-None type is expected",
    "none-arg": "None Argument: Passing `None` where a non-None type is expected",
    "import-not-found": "Import Not Found: Missing or uninstalled import",
    "override": "Override: Subclass method violates Liskov substitutability principle",
    "func-returns-value": "Function Returns Value: Function returns `None` but a value is expected",
    "has-type": "Has Type: Cannot determine type of a variable",
    "return": "Return: Missing return statement",
    "name-match": "Name Match: Argument type mismatch in a namedtuple",
    "truthy-function": "Truthy Function: Function is always true in a boolean context (conditional)",
}


def classify_errors(errors: list[Error]) -> dict[str, list[Error]]:
    classified_errors: dict[str, list[Error]] = defaultdict(list)
    for error in errors:
        if error["severity"] != "error":
            continue
        category = CATEGORIES.get(error["code"], "Unknown")
        if category == "Unknown":
            print(f"Unknown: {error}")
        classified_errors[category].append(error)
    return classified_errors


def summarize_errors(
    classified_errors: dict[str, list[Error]]
) -> tuple[list[tuple[str, str, int, str]], int]:
    total_errors = sum(len(errors) for errors in classified_errors.values())
    summary: list[tuple[str, str, int, str]] = []

    for category, errors in classified_errors.items():
        count = len(errors)
        percentage = (count / total_errors) * 100
        summary.append(
            (category, f"{errors[0]['code']}", count, f"{percentage:.2f}%")
        )

    # Sort by percentage in descending order
    summary.sort(key=lambda x: float(x[3][:-1]), reverse=True)

    return summary, total_errors


def display_summary(
    summary: list[tuple[str, str, int, str]], total_errors: int
) -> None:
    console = Console()
    table = Table(title="Mypy Error Summary")
    table.add_column("Category", justify="left", style="cyan", no_wrap=True)
    table.add_column("Error Code", justify="left", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Percentage", justify="right", style="green")

    for category, error_code, count, percentage in summary:
        table.add_row(
            category, f"[bold]{error_code}[/bold]", str(count), percentage
        )

    console.print(table)
    console.print(f"\nTotal Errors: {total_errors}\n")


def display_summary_markdown(
    summary: list[tuple[str, str, int, str]], total_errors: int
) -> None:
    # Generate markdown table
    markdown_table = "# Mypy Error Summary\n\n"
    markdown_table += "| Category | Error Code | Count | Percentage |\n"
    markdown_table += "|---|---|---|---|\n"

    for category, error_code, count, percentage in summary:
        markdown_table += (
            f"| {category} | {error_code} | {count} | {percentage} |\n"
        )

    markdown_table += f"\n**Total Errors**: {total_errors}\n"

    print(markdown_table)


def display_summary_csv(
    summary: list[tuple[str, str, int, str]], total_errors: int
) -> None:
    # Generate CSV-style output
    csv_table = "Category,Error Code,Count,Percentage\n"

    for category, error_code, count, percentage in summary:
        csv_table += f"{category},{error_code},{count},{percentage}\n"

    csv_table += f"\nTotal Errors,{total_errors},\n"
    print(csv_table)


@click.command()
@click.option("--markdown", is_flag=True, default=False)
@click.option("--csv", is_flag=True, default=False)
@click.argument(
    "file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
def main(file: str, markdown: bool, csv: bool) -> None:
    # MyPy output is in jsonl format (JSON Lines); parse accordingly.
    errors: list[Error] = []
    with open(file) as f:
        for line in f:
            try:
                errors.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # ignore non-JSON lines

    classified_errors = classify_errors(errors)

    summary, total_errors = summarize_errors(classified_errors)

    if csv:
        display_summary_csv(summary, total_errors)
    elif markdown:
        display_summary_markdown(summary, total_errors)
    else:
        display_summary(summary, total_errors)


if __name__ == "__main__":
    main()
