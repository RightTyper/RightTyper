import datetime
import json
import logging
import sys

logger = logging.getLogger(__name__.split('.')[0])
logger.setLevel(logging.INFO)
logger.propagate = False    # excludes other packages' messages

_handler = logging.FileHandler(logger.name + '.log')
_handler.setFormatter(logging.Formatter("[%(filename)s:%(lineno)s] %(message)s"))

logger.addHandler(_handler)

# Sampling evaluation logger — no handlers until init_sampling_log() is called,
# so no file is created unless --log-sampling or --eval-sampling is passed.
sampling_logger = logger.getChild('sampling')
sampling_logger.propagate = False

_sampling_handler: logging.FileHandler | None = None
_sampling_summary: dict = {}

def init_sampling_log() -> None:
    """Activate sampling log output. Only call when --log-sampling or --eval-sampling is set."""
    global _sampling_handler, _sampling_summary
    from righttyper.options import run_options

    _sampling_handler = logging.FileHandler(logger.name + '-sampling.jsonl', mode='a')
    _sampling_handler.setFormatter(logging.Formatter('%(message)s'))
    sampling_logger.addHandler(_sampling_handler)

    _sampling_summary = {
        'total_observations': 0,
        'eval_observations': 0,
        'perfect_recall': 0,
    }

    # Write run-start marker
    sampling_logger.info(json.dumps({
        '_run_start': True,
        'timestamp': datetime.datetime.now().isoformat(),
        'command': ' '.join(sys.orig_argv),
        'config': {
            'eval_sampling': run_options.eval_sampling,
            'container_small_threshold': run_options.container_small_threshold,
            'container_max_samples': run_options.container_max_samples,
            'container_type_threshold': run_options.container_type_threshold,
            'container_min_samples': run_options.container_min_samples,
            'container_check_probability': run_options.container_check_probability,
            'container_sample_range': run_options.container_sample_range,
        },
    }))


def update_sampling_summary(record: dict) -> None:
    """Update running summary counters from a logged record. No-op if init was never called."""
    if not _sampling_summary:
        return
    _sampling_summary['total_observations'] += 1
    if 'recall' in record:
        _sampling_summary['eval_observations'] += 1
        if record['recall'] == 1.0:
            _sampling_summary['perfect_recall'] += 1


def finalize_sampling_log() -> None:
    """Write summary and close the sampling log. No-op if init was never called."""
    global _sampling_handler
    if _sampling_handler is None:
        return

    sampling_logger.info(json.dumps({
        '_run_end': True,
        'timestamp': datetime.datetime.now().isoformat(),
        'summary': _sampling_summary,
    }))

    sampling_logger.removeHandler(_sampling_handler)
    _sampling_handler.close()
    _sampling_handler = None
