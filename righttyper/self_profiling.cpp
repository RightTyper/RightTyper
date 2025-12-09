#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <atomic>
#include <thread>
#include <chrono>

static constexpr double ALPHA = 0.4;        // for exponential smoothing
static constexpr int PERIOD = 250;          // compute overhead every N samples
static constexpr double TIMER_INTERVAL = 0.002;

namespace py = pybind11;

static double exp_smooth(double value, double previous) {
    return (previous < 0.0) ? value : ALPHA * value + (1.0 - ALPHA) * previous;
}

struct State {
    // atomic counter: number of "active instrumentation" contexts
    std::atomic<long long> _instr_active_count{0};

    // sampling window
    long _sample_count_instrumentation = 0;
    long _sample_count_total = 0;
    double _overhead = -1.0;  // sentinel value until first measured

    bool _configured = false;

    // Python lists to store history
    std::vector<long long> _hist_samples_instr;
    std::vector<long long> _hist_samples_total;
    std::vector<double>    _hist_overhead;
    std::vector<bool>      _hist_restarted;

    // External dependencies provided by user code
    py::object _disabled_code;    // set; we call .clear()
    py::object _restart_callable; // callable to (re)start events
    py::object _unwind_handler;

    // Cached settings (read once from 'run_options' in configure)
    double _target_overhead_threshold = 0.05; // proportion, e.g., 0.025 for 2.5%
    bool _save_profiling = false;

    // Background timer
    std::atomic<bool> _timer_running{false};
    std::thread _timer_thread;
    double _interval_sec = TIMER_INTERVAL;


    // Configures self-profiling
    void configure(const py::object& run_options, const py::set& disabled_code, const py::function& restart_callable) {
        if (run_options.is_none())
            throw std::runtime_error("configure: 'run_options' must not be None.");

        try {
            double target_percent = py::cast<double>(run_options.attr("target_overhead"));
            _target_overhead_threshold = target_percent / 100.0;
        } catch (const py::error_already_set&) {
            throw std::runtime_error("run_options.target_overhead must be a float (percent).");
        }

        _save_profiling = !(run_options.attr("save_profiling")).is_none();

        if (disabled_code.is_none())
            throw std::runtime_error("configure: 'disabled_code' must not be None.");
        _disabled_code = disabled_code;

        if (restart_callable.is_none())
            throw std::runtime_error("configure: 'restart_callable' must not be None.");
        _restart_callable = restart_callable;

        _configured = true;
    }


    // Called to indicate entering self-instrumentation
    void enter_instrumentation() {
        _instr_active_count.fetch_add(1, std::memory_order_relaxed);
    }


    // Called to indicate exiting self-instrumentation
    void exit_instrumentation() {
        long long new_val = _instr_active_count.fetch_sub(1, std::memory_order_relaxed) - 1;
        if (new_val < 0) {
            // clamp to zero and warn
            _instr_active_count.store(0, std::memory_order_relaxed);
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "exit_instrumentation() called when counter was already zero; clamping.", 1);
        }
    }


    // Called regularly by timer thread to self-profile
    void self_profile() {
        if (!_configured) return;

        _sample_count_total += 1;

        if (_instr_active_count.load(std::memory_order_relaxed) > 0) {
            _sample_count_instrumentation += 1;
        }

        if ((_sample_count_total % PERIOD) == 0) {
            const double this_period =
                    static_cast<double>(_sample_count_instrumentation)
                        / static_cast<double>(_sample_count_total);

            _overhead = exp_smooth(this_period, _overhead);

            const bool restart = (_overhead <= _target_overhead_threshold);
            if (restart) {
                py::gil_scoped_acquire gil;  // ensure GIL for Python interactions

                // TODO maybe combine these in a single callable
                _disabled_code.attr("clear")();
                _restart_callable();
            }

            if (_save_profiling) {
                _hist_samples_instr.push_back(_sample_count_instrumentation);
                _hist_samples_total.push_back(_sample_count_total);
                _hist_overhead.push_back(_overhead);
                _hist_restarted.push_back(restart);
            }

            _sample_count_instrumentation = _sample_count_total = 0;
        }
    }


    // Returns collected self-profiling history, if any
    py::dict get_history() {
        py::list a, b, c, d;
        for (auto v : _hist_samples_instr)  a.append(v);
        for (auto v : _hist_samples_total)  b.append(v);
        for (auto v : _hist_overhead)       c.append(v);
        for (auto v : _hist_restarted)      d.append(py::bool_(v));

        py::dict out;
        out["samples_instrumentation"] = std::move(a);
        out["samples_total"] = std::move(b);
        out["overhead"] = std::move(c);
        out["restarted"] = std::move(d);
        return out;
    }


    // Starts the timer thread
    void start_timer(double interval) {
        _interval_sec = interval > 0 ? interval : TIMER_INTERVAL;
        if (_timer_running.exchange(true, std::memory_order_acq_rel)) {
            return;  // already running
        }

        // Note that if multiple threads calling start/stop are possible,
        // we need a mutex around _timer_thread
        _timer_thread = std::thread([this]() {
            using namespace std::chrono;
            const auto tick = duration<double>(_interval_sec);
            while (_timer_running.load(std::memory_order_acquire)) {
                this->self_profile();
                std::this_thread::sleep_for(tick);
            }
        });
    }


    // Stops the timer thread
    void stop_timer() {
        if (_timer_running.exchange(false, std::memory_order_acq_rel)) {
            return;  // already stopped
        }

        if (_timer_thread.joinable()) {
            // don't hold the GIL while joining, as the thread may be trying to get it
            py::gil_scoped_release nogil;
            _timer_thread.join();
        }
    }

    void set_unwind_handler(const py::function& unwind_handler) {
        _unwind_handler = unwind_handler;
    }

    void unwind_handler(const py::object& code, int offset, const py::object& exception) {
        // Since PY_UNWIND can't be disabled, we do some filtering for relevant events here.
        if (_configured && !_unwind_handler.is_none()) {
            py::set disabled = py::reinterpret_borrow<py::set>(_disabled_code);
            uintptr_t id = reinterpret_cast<uintptr_t>(code.ptr());
            if (!disabled.contains(id))
                _unwind_handler(code, offset, exception);
        }
    }


    // Called from Python to clean up
    void cleanup() {
        _disabled_code = py::none();
        _restart_callable = py::none();
        _unwind_handler = py::none();
        stop_timer(); // must be last: releases the GIL
    }


    ~State() {
        // Safety on interpreter/module teardown
        try { stop_timer(); } catch (...) {}
    }
};

static State G;


PYBIND11_MODULE(self_profiling, m) {
    m.doc() = "C++ extension for self-profiling instrumentation";

    m.def("configure",
          [](py::object run_options, py::set disabled_code, py::function restart_callable) {
              G.configure(run_options, disabled_code, restart_callable);
          },
          py::arg("run_options"),
          py::arg("disabled_code"),
          py::arg("restart_callable")
    );

    m.def("enter_instrumentation", []() { G.enter_instrumentation(); },
          "Called to indicate entering instrumentation."
    );
    m.def("exit_instrumentation", []() { G.exit_instrumentation(); },
          "Called to indicate exiting instrumentation."
    );

    m.def("start", [](double interval_sec) { G.start_timer(interval_sec); },
          py::arg("interval_sec") = TIMER_INTERVAL,
          "Starts self-profiling."
    );
    m.def("stop", []() { G.stop_timer(); },
          "Stops self-profiling."
    );

    m.def("get_history", []() { return G.get_history(); }, "Returns self-profiling history.");

    m.def("set_unwind_handler", [](py::function handler) { G.set_unwind_handler(handler); });
    m.def("unwind_handler",
          [](py::object code, int offset, py::object exception) -> py::object {
              G.unwind_handler(code, offset, exception);
              return py::none();
          }
    );

    // If we don't clean up our references to Python objects while Python is still alive,
    // we may get a SIGSEGV in ~State() as it tries to reclaim memory.
    // Use a Capsule object for this:
    m.add_object("_teardown", py::capsule(&G, [](void* ptr) {
            try {
                py::gil_scoped_acquire gil;
                static_cast<State*>(ptr)->cleanup();
            } catch (...) {
                // swallow any errors during teardown
            }
        })
    );
    
}
