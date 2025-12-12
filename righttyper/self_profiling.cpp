#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <atomic>
#include <thread>
#include <chrono>

namespace py = pybind11;

class __attribute__((visibility("hidden"))) State {
    py::object sysmon() {
        return py::module_::import("sys").attr("monitoring");
    }

    // atomic counter: number of "active instrumentation" contexts
    std::atomic<long> _instr_count{0};
    std::atomic<long> _instr_python_count{0};

    // Python lists to store history
    std::vector<long> _hist_instr;
    std::vector<long> _hist_instr_python;
    std::vector<bool> _hist_restarted;

    // Sets of disabled code objects
    py::object _disabled = py::set();
    py::object _permanently_disabled = py::set();

    // provided by configure()
    double _reenable_interval = 0.5; // seconds
    int _reenable_max_calls = 1;     // must fall below this to reenable
    bool _configured = false;

public:
    enum Event {
        START, YIELD, RETURN, UNWIND,
        COUNT // dummy for length
    };

private:
    py::object _handler[COUNT];
    py::object _restart_events = sysmon().attr("restart_events");
    py::object _sysmon_disable = sysmon().attr("DISABLE");

    // Background timer
    std::atomic<bool> _timer_running{false};
    std::thread _timer_thread;


public:
    State() {
        for (py::object& h: _handler)
            h = py::none();
    }

    // Configures self-profiling
    void configure(const py::object& run_options) {
        if (run_options.is_none())
            throw std::runtime_error("configure: 'run_options' must not be None.");

        _reenable_interval = run_options.attr("reenable_interval").cast<double>();
        _reenable_max_calls = run_options.attr("reenable_max_calls").cast<int>();
        _configured = true;
    }

    // Updates counters and saves history
    bool update() {
        auto instr_count = _instr_count.exchange(0, std::memory_order_relaxed);
        auto instr_python_count = _instr_python_count.exchange(0, std::memory_order_relaxed);

        bool restart = (
            instr_python_count < _reenable_max_calls
            && py::len(_disabled) > py::len(_permanently_disabled) // anything disabled since the last restart?
        );

        _hist_instr.push_back(instr_count);
        _hist_instr_python.push_back(instr_python_count);
        _hist_restarted.push_back(restart);
        return restart;
    }

    // Called regularly by timer thread to self-profile
    void self_profile() {
        bool should_restart = update();
        if (should_restart) {
            py::gil_scoped_acquire gil;  // ensure GIL for Python interactions

            _disabled = py::set(_permanently_disabled);
            _restart_events();
        }
    }

    // Returns collected self-profiling history, if any
    py::dict get_history() {
        py::dict out;
        out["instrumentation"] = py::cast(_hist_instr);
        out["instrumentation_python"] = py::cast(_hist_instr_python);
        out["restarted"] =  [this]() {
            py::list l;
            for (bool b: _hist_restarted)
                l.append(py::bool_(b));
            return l;
        }();
        out["reenable_interval"] = py::cast(_reenable_interval);
        out["reenable_max_calls"] = py::cast(_reenable_max_calls);
        return out;
    }

    // Starts the timer thread
    void start_timer() {
        if (_timer_running.exchange(true, std::memory_order_acq_rel)) {
            return;  // already running
        }

        // Note that if multiple threads calling start/stop are possible,
        // we need a mutex around _timer_thread
        _timer_thread = std::thread([this]() {
            using namespace std::chrono;
            const auto tick = duration<double>(_reenable_interval);
            while (_timer_running.load(std::memory_order_acquire)) {
                this->self_profile();
                std::this_thread::sleep_for(tick);
            }

            this->update(); // for any partial period
        });
    }

    // Stops the timer thread
    void stop_timer() {
        if (!_timer_running.exchange(false, std::memory_order_acq_rel)) {
            return;  // already stopped
        }

        if (_timer_thread.joinable()) {
            // don't hold the GIL while joining, as the thread may be trying to get it
            py::gil_scoped_release nogil;
            _timer_thread.join();
        }
    }

    // Disables code until restart
    void disable(const py::object code) {
         _disabled.attr("add")(code);
    }

    // Disables code permanently
    void permanently_disable(const py::object code) {
         _permanently_disabled.attr("add")(code);
         _disabled.attr("add")(code);
    }

    // Sets the given handler
    void set_handler(Event event, const py::function& handler) {
        _handler[event] = handler;
    }

    // Handles a sys.monitoring event
    py::object handler(Event event, py::args& args) {
        _instr_count.fetch_add(1, std::memory_order_relaxed);

        if (_disabled.contains(args[0]))
            return event != UNWIND ? _sysmon_disable : py::none();

        if (!_handler[event].is_none()) {
            _instr_python_count.fetch_add(1, std::memory_order_relaxed);
            return _handler[event](*args);
        }

        return py::none();
    }

    // Called from Python to clean up
    void cleanup() {
        for (py::object& h: _handler)
            h = py::none();

        _permanently_disabled = py::none();
        _disabled = py::none();

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

    m.def("configure", [](py::object run_options) {G.configure(run_options); });

    m.def("disable", [](py::object code) {G.disable(code);});
    m.def("permanently_disable", [](py::object code) {G.permanently_disable(code);});

    m.def("start", []() {G.start_timer();});
    m.def("stop", []() {G.stop_timer();});

    m.def("get_history", []() {return G.get_history();});

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
