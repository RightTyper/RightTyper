#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <atomic>
#include <thread>
#include <chrono>

static constexpr double ALPHA = 0.4;        // for exponential smoothing
static constexpr int PERIOD = 500;          // compute overhead every N samples
static constexpr double TIMER_INTERVAL = 0.001;

static const std::vector<float> MODEL = {
    1.2850532814104125, 3.9667904371269294, -1.1454537705064482e-05
};

namespace py = pybind11;

static double exp_smooth(double value, double previous) {
    return (previous < 0.0) ? value : ALPHA * value + (1.0 - ALPHA) * previous;
}

class InstrGuard {
    std::atomic<long long>& _counter;

public:
    InstrGuard(std::atomic<long long>& counter) : _counter(counter) {
        _counter.fetch_add(1, std::memory_order_relaxed);
    }

    ~InstrGuard() {
        long long new_val = _counter.fetch_sub(1, std::memory_order_relaxed) - 1;
        if (new_val < 0) {
            // clamp to zero and warn
            _counter.store(0, std::memory_order_relaxed);
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "exit_instrumentation() called when counter was already zero; clamping.", 1);
        }
    }
};

class __attribute__((visibility("hidden"))) State {
    py::object sysmon() {
        return py::module_::import("sys").attr("monitoring");
    }

    // atomic counter: number of "active instrumentation" contexts
    std::atomic<long long> _instr_active_count{0};

    // sampling window
    long _sample_count_instr = 0;
    long _sample_count_total = 0;

    bool _configured = false;

    // Python lists to store history
    long _prev_count_instr = 0;
    long _prev_count_total = 0;
    std::vector<long long> _hist_samples_instr;
    std::vector<long long> _hist_samples_total;
    std::vector<double>    _hist_overhead;
    std::vector<bool>      _hist_restarted;

    // External dependencies provided by user code
    py::object _disabled_code = py::none(); // set; we call .clear()

public:
    enum Event {
        START, YIELD, RETURN, UNWIND,
        COUNT // dummy for length
    };

private:
    py::object _handler[COUNT];
    py::object _restart_events = sysmon().attr("restart_events");
    py::object _sysmon_disable = sysmon().attr("DISABLE");

    // Cached settings (read once from 'run_options' in configure)
    double _target_overhead_threshold = 1.05;

    // Background timer
    std::atomic<bool> _timer_running{false};
    std::thread _timer_thread;
    double _interval_sec = TIMER_INTERVAL;


public:
    State() {
        for (py::object& h: _handler)
            h = py::none();
    }

    // Configures self-profiling
    void configure(const py::object& run_options, const py::set& disabled_code) {
        if (run_options.is_none())
            throw std::runtime_error("configure: 'run_options' must not be None.");

        try {
            double target_percent = py::cast<double>(run_options.attr("target_overhead"));
            _target_overhead_threshold = 1.0 + target_percent / 100.0;
        } catch (const py::error_already_set&) {
            throw std::runtime_error("run_options.target_overhead must be a float (percent).");
        }

        _disabled_code = disabled_code;
        _configured = true;
    }


    bool calculate_overhead() {
        if (_sample_count_total) {
            auto overhead = static_cast<double>(_sample_count_instr)
                          / static_cast<double>(_sample_count_total);
            overhead = MODEL[0] + MODEL[1]*overhead + MODEL[2]*_sample_count_total;

//            _overhead = exp_smooth(period_overhead, _overhead);
            const bool restart = (overhead <= _target_overhead_threshold);

            // save history
            _hist_samples_instr.push_back(_sample_count_instr - _prev_count_instr);
            _prev_count_instr = _sample_count_instr;
            _hist_samples_total.push_back(_sample_count_total - _prev_count_total);
            _prev_count_total = _sample_count_total;
            _hist_overhead.push_back(overhead);
            _hist_restarted.push_back(restart);

            return restart;
        }

        return false;
    }


    // Called regularly by timer thread to self-profile
    void self_profile() {
        if (!_configured) return;

        // sample instrumentation activity
        _sample_count_total += 1;
        if (_instr_active_count.load(std::memory_order_relaxed) > 0) {
            _sample_count_instr += 1;
        }

        if ((_sample_count_total % PERIOD) == 0) {
            bool should_restart = calculate_overhead();
            if (should_restart) {
                py::gil_scoped_acquire gil;  // ensure GIL for Python interactions

                // TODO maybe combine these in a single callable
                _disabled_code.attr("clear")();
                _restart_events();
            }
        }
    }


    // Returns collected self-profiling history, if any
    py::dict get_history() {
        py::dict out;
        out["samples_instrumentation"] = py::cast(_hist_samples_instr);
        out["samples_total"] = py::cast(_hist_samples_total);
        out["overhead"] = py::cast(_hist_overhead);
        out["restarted"] =  [this]() {
            py::list l;
            for (bool b: _hist_restarted)
                l.append(py::bool_(b));
            return l;
        }();
        out["model"] = py::cast(MODEL);
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

            // calculate a last time to save any partial period
            this->calculate_overhead();
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

    void set_handler(Event event, const py::function& handler) {
        _handler[event] = handler;
    }

    py::object handler(Event event, py::args& args) {
        InstrGuard g(_instr_active_count);

        if (_configured && !_handler[event].is_none()) {
            py::set disabled = py::reinterpret_borrow<py::set>(_disabled_code);
            if (disabled.contains(args[0]))
                return event != UNWIND ? _sysmon_disable : py::none();

            return _handler[event](*args);
        }

        return py::none();
    }

    // Called from Python to clean up
    void cleanup() {
        _disabled_code = py::none();

        for (py::object& h: _handler)
            h = py::none();

        _restart_events = py::none();
        stop_timer(); // must be last: releases the GIL
    }


    ~State() {
        // Safety on interpreter/module teardown
        try { stop_timer(); } catch (...) {}
    }
};


PYBIND11_MODULE(self_profiling, m) {
    static State G;

    m.doc() = "C++ extension for self-profiling instrumentation";

    m.def("configure",
          [](py::object run_options, py::set disabled_code) {
              G.configure(run_options, disabled_code);
          },
          py::arg("run_options"),
          py::arg("disabled_code")
    );

    m.def("start", [](double interval_sec) { G.start_timer(interval_sec); },
          py::arg("interval_sec") = TIMER_INTERVAL,
          "Starts self-profiling."
    );
    m.def("stop", []() { G.stop_timer(); },
          "Stops self-profiling."
    );

    m.def("get_history", []() { return G.get_history(); }, "Returns self-profiling history.");

    m.def("set_start_handler", [](py::function handler) { G.set_handler(G.START, handler); });
    m.def("start_handler", [](py::args args) -> py::object {return G.handler(G.START, args);});

    m.def("set_yield_handler", [](py::function handler) { G.set_handler(G.YIELD, handler); });
    m.def("yield_handler", [](py::args args) -> py::object {return G.handler(G.YIELD, args);});

    m.def("set_return_handler", [](py::function handler) { G.set_handler(G.RETURN, handler); });
    m.def("return_handler", [](py::args args) -> py::object {return G.handler(G.RETURN, args);});

    m.def("set_unwind_handler", [](py::function handler) { G.set_handler(G.UNWIND, handler); });
    m.def("unwind_handler", [](py::args args) -> py::object {return G.handler(G.UNWIND, args);});

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
