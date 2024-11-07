import textwrap
import subprocess
import sys
from pathlib import Path


def test_issue_22(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def extracted_function(A):
            return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or all(A[i] >=
                A[i + 1] for i in range(len(A) - 1))

        def optimized(A):
            return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) and all(A[i] >=
                A[i + 1] for i in range(len(A) - 1))

        def main():
            assert extracted_function([6, 5, 4, 4]) == True
            assert extracted_function([1, 2, 2, 3]) == True
            assert extracted_function([1, 3, 2]) == False

        if __name__ == '__main__':
            main()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    
    assert "def extracted_function(A: list[int]) -> bool" in Path("t.py").read_text()
