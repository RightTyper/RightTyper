# RightTyper

RightTyper is a Python tool that generates types for your function
arguments and return values. It is inspired by and produces much the
same results as Instagram's `monkeytype`.  At the same time,
RightTyper's approach ends up being more flexible and up to fifty
times faster. RightTyper lets your code run at nearly full speed with
almost no memory overhead. As a result, you won't experience slow
downs in your code or large memory consumption while using it,
allowing you to integrate it with your standard tests and development
process.

## Installation

To install the latest version of RightTyper from its repository,
just use `pip` as shown below:

```bash
python3 -m pip install git+https://github.com/RightTyper/righttyper
```

## `righttyper`: high performance

In the below example drawn from the pyperformance benchmark suite,
`monkeytype` runs **40x slower** than the original program or when
running with `righttyper` (which runs under 3% slower).

```bash
% python3 bm_mdp          
Time elapsed:  6.106977417017333
% righttyper bm_mdp
Time elapsed:  6.299191833997611
% monkeytype run bm_mdp
Time elapsed:  184.57902495900635
# actual time elapsed was 275 seconds, spent post-processing
```

# `righttyper`: low memory consumption

With `monkeytype`, this program also consumes 5GB of RAM; the original
consumes just 21MB. That's an over **200x** increase in memory
consumption. `monkeytype` also leaves behind a 3GB SQLite file.

By contrast, `righttyper`'s memory consumption is just a small
increment over the original program: it consumes about 24MB, just 15%
more.

_NOTE: this is an alpha release and is not production ready._

## Requirements

- Python 3.12 or higher

## Usage

To use RightTyper, simply run your script with `righttyper` instead of `python3`:

```bash
righttyper your_script.py [args...]
```

This will execute `your_script.py` with RightTyper's monitoring enabled. The type signatures of all functions will be recorded and output to a file named `righttyper.out`. Each line represents a function signature in the following format:

```
filename:def function_name(arg1: arg1_type, arg2: arg2_type, ...) -> return_type
```

## How it works

Monkeytype is slow because it uses Python's `setprofile` functionality
to track every single function call and return, gathers types for all
arguments and the return value, and then writes these into a SQLite
database.

By contrast, RightTyper leverages Python 3.12's new `sys.monitoring`
mechanism to allow it to add and remove type checking. It always
checks the very first invocation and exit of every function, and then
disables monitoring. It re-enables monitoring periodically with
decreasing frequency. Of course, sampling makes it unlikely to capture
rare events, which may correspond to different types (e.g., conditions
that result in `None` being returned). RightTyper avoids this pitfall
by employing a heuristic that reactivates monitoring when events occur
that we expect to be highly correlated with the use of different
types. Specifically, RightTyper re-enables monitoring in functions
when the program executes branches and lines for the first time. This
approach ensures that RightTyper finds any type differences that are
based on control flow.
