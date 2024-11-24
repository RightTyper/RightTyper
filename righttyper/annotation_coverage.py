import ast
import io
import os
import warnings

import click

from righttyper.righttyper_utils import TOOL_NAME

# Suppress SyntaxWarning during AST parsing
warnings.filterwarnings("ignore", category=SyntaxWarning)

partially_annotated: list[tuple[str, str, int]] = []
not_annotated: list[tuple[str, str, int]] = []


class FullyQualifiedNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.current_class: str|None = None
        self.qualified_names: dict[
            ast.FunctionDef | ast.AsyncFunctionDef,
            str,
        ] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self,
        node: ast.FunctionDef|ast.AsyncFunctionDef,
    ) -> None:
        if self.current_class:
            qualified_name = f"{self.current_class}.{node.name}"
        else:
            qualified_name = node.name
        self.qualified_names[node] = qualified_name
        self.generic_visit(node)


def generate_fully_qualified_names_dict(
    tree: ast.AST,
) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, str]:
    collector = FullyQualifiedNameCollector()
    collector.visit(tree)
    return collector.qualified_names


def parse_python_file(
    file_path: str,
) -> list[int]:
    fully_annotated_count = 0
    partially_annotated_count = 0
    not_annotated_count = 0

    import sys
    
    with open(file_path, "r") as file:
        try:
            tree = ast.parse(file.read(), filename=file_path)
        except SyntaxError:
            tree = ast.parse("", filename=file_path)
        except UnicodeDecodeError:
            tree = ast.parse("", filename=file_path)

    # Dynamically adapt the recursion limit as needed
    old_recursion_limit = sys.getrecursionlimit()
    while True:
        try:
            qualified_names = generate_fully_qualified_names_dict(tree)
            break
        except RecursionError:
            # Try again but increasing the previous limit twofold
            sys.setrecursionlimit(sys.getrecursionlimit() * 2)

    # Restore the original recursion limit.
    sys.setrecursionlimit(old_recursion_limit)
    
    # Function to check if a node has annotations
    def has_annotations(node: ast.arg) -> bool:
        return hasattr(node, "annotation") and node.annotation is not None

    # Function to recursively search for functions in the AST
    def search_functions(node: ast.AST) -> None:
        nonlocal fully_annotated_count, partially_annotated_count, not_annotated_count

        if isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ):
            annotations_count = 0
            total_to_annotate = 1  # Count the return here, then the args

            if node.args.args:
                for index, arg in enumerate(node.args.args):
                    # Modest unsoundness:
                    # We assume that if the first argument is self or cls,
                    # it is actually in a method / classmethod
                    if index == 0 and arg.arg in ["self", "cls"]:
                        continue
                    total_to_annotate += 1
                    annotations_count += has_annotations(arg)

            # Check if there is a return type annotation
            if node.returns:
                annotations_count += 1

            entry = (
                file.name,
                qualified_names[node],
                node.lineno,
            )
            if annotations_count == total_to_annotate:
                fully_annotated_count += 1
            elif annotations_count > 0:
                partially_annotated.append(entry)
                partially_annotated_count += 1
            else:
                not_annotated.append(entry)
                not_annotated_count += 1

        for child_node in ast.iter_child_nodes(node):
            search_functions(child_node)

    search_functions(tree)

    return [
        fully_annotated_count,
        partially_annotated_count,
        not_annotated_count,
    ]


def analyze_directory(
    directory: str, cache: dict[str, list[int]]
) -> list[int]:
    if directory in cache:
        return cache[directory]

    directory_summary: list[int] = [
        0,
        0,
        0,
    ]  # [fully annotated, partially annotated, not annotated]

    # Check if the directory argument is a single file
    files: list[str] = []
    dirs: list[str] = []
    root = ""
    if os.path.isfile(directory):
        files = [os.path.basename(directory)]
        dirs = []
        root = os.path.dirname(directory)
    else:
        for root, dirs, files in os.walk(directory):
            break  # We only need the top-level directory content

    for file_name in files:
        if file_name.endswith(".py"):
            file_path = os.path.join(root, file_name)
            (
                fully_annotated_count,
                partially_annotated_count,
                not_annotated_count,
            ) = parse_python_file(file_path)
            directory_summary[0] += fully_annotated_count
            directory_summary[1] += partially_annotated_count
            directory_summary[2] += not_annotated_count
            # Update file-level summary in cache
            cache[file_path] = [
                fully_annotated_count,
                partially_annotated_count,
                not_annotated_count,
            ]

    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        subdir_summary = analyze_directory(dir_path, cache)
        directory_summary[0] += subdir_summary[0]  # fully annotated
        directory_summary[1] += subdir_summary[1]  # partially annotated
        directory_summary[2] += subdir_summary[2]  # not annotated

    cache[directory] = directory_summary
    return directory_summary


def analyze_all_directories(
    directory: str,
) -> dict[str, list[int]]:
    cache: dict[str, list[int]] = {}
    analyze_directory(directory, cache)
    return cache


def print_directory_summary(summary: dict[str, list[int]]) -> None:
    from rich.console import Console
    from rich.table import Table

    headers = [
        "Directory",
        "Fully\nannotated",
        "Partially\nannotated",
        "Unannotated",
        "% Fully\n annotated",
    ]
    data = []
    for (
        directory,
        counts,
    ) in (
        summary.items()
    ):  # sorted(summary.items(), key=lambda x: x[1][2]):  # Sort by number of not annotated functions
        if os.path.isfile(directory):  # Check if the path is a file
            continue
        total_functions = sum(counts)
        if total_functions > 0:
            percentage_annotated = (counts[0] / total_functions) * 100
        else:
            continue
        data.append(
            [
                directory,
                f"{counts[0]}",
                f"{counts[1]}",
                f"{counts[2]}",
                f"{percentage_annotated:6.2f}%",
            ]
        )

    summary_row = data[-1]
    print(
        f"{summary_row[3]} unannotated, {summary_row[2]} partially annotated, {summary_row[1]} fully annotated ({summary_row[4]})"
    )

    table = Table(
        show_header=True,
        header_style="bold magenta",
    )
    for header in headers:
        table.add_column(header)

    for row in data:
        table.add_row(*row)

    output_buffer = io.StringIO()
    console = Console(file=output_buffer, width=255, record=True)
    console.print(table)
    html_output = console.export_html()

    # Save the HTML output to a file
    with open(f"{TOOL_NAME}-coverage.html", "w") as file:
        file.write(html_output)
    print(f"Report saved in {TOOL_NAME}-coverage.html")


def print_file_summary(summary: dict[str, list[int]]) -> None:
    from rich.console import Console
    from rich.table import Table

    headers = [
        "File",
        "Fully\nannotated",
        "Partially\nannotated",
        "Unannotated",
        "% Fully\nannotated",
    ]
    data = []
    # print(summary.items())
    for file_path, counts in sorted(
        summary.items()
    ):  # , key=lambda x: sum(x[1:])): # x[1][2]):  # Sort by number of not annotated functions
        if os.path.isfile(file_path):  # Check if the path is a file
            total_functions = sum(counts)
            total_annotated_functions = total_functions - counts[2] - counts[1]
            if total_functions > 0:
                percentage_annotated = (
                    total_annotated_functions / total_functions
                ) * 100
            else:
                continue
            data.append(
                [
                    file_path,
                    f"{counts[0]}",
                    f"{counts[1]}",
                    f"{counts[2]}",
                    f"{percentage_annotated:6.2f}%",
                ]
            )

    table = Table(
        show_header=True,
        header_style="bold magenta",
    )
    for header in headers:
        table.add_column(header)

    for row in data:
        table.add_row(*row)

    table.add_section()
    c = [0, 0, 0]
    for file_path, counts in summary.items():
        if not os.path.isfile(file_path):  # Check if the path is a file
            continue
        # print(f"DIRECTORY {file_path} COUNTS {counts}")
        for i, value in enumerate(counts):
            c[i] += value
    total_functions = sum(c)
    if total_functions > 0:
        percentage_annotated = (c[0] / total_functions) * 100
    else:
        percentage_annotated = 0
    summary_row = [
        "Total:",
        f"{c[0]}",
        f"{c[1]}",
        f"{c[2]}",
        f"{percentage_annotated:6.2f}%",
    ]
    table.add_row(*summary_row)

    print(
        f"{summary_row[3]} unannotated, {summary_row[2]} partially annotated, {summary_row[1]} fully annotated ({summary_row[4]})"
    )

    output_buffer = io.StringIO()
    console = Console(file=output_buffer, width=255, record=True)
    console.print(table)
    html_output = console.export_html()

    # Save the HTML output to a file
    with open(f"{TOOL_NAME}-coverage.html", "w") as file:
        file.write(html_output)
    print(f"Report saved in {TOOL_NAME}-coverage.html")


def print_annotation_summary() -> None:
    items = [partially_annotated, not_annotated]
    item_titles = [
        "Partially annotated",
        "Not annotated",
    ]
    for i in range(len(items)):
        if items[i]:
            items[i] = sorted(
                items[i],
                key=lambda x: (x[0], x[2]),
            )
            print(f"{item_titles[i]}:")
            print("-" * (len(item_titles[i]) + 1))
            while items[i]:
                fname = items[i][0][0]
                print(f"{fname}:")
                while items[i] and fname == items[i][0][0]:
                    rec = items[i].pop(0)
                    print(f"  {rec[2]:5}:{rec[1]}")
            print()


@click.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "-l",
    "--level",
    type=click.Choice(["directory", "file"]),
    default="directory",
    help="Level of statistics to report: directory or file.",
)
def main(directory: str, level: str) -> None:
    if level == "directory":
        directory_summary = analyze_all_directories(directory)
        print_directory_summary(directory_summary)
    elif level == "file":
        file_summary = analyze_all_directories(directory)
        print_file_summary(file_summary)
    print_annotation_summary()


if __name__ == "__main__":
    main()
