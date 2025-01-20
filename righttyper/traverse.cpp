#define PY_SSIZE_T_CLEAN
extern "C" {
    #include <Python.h>
}
#include <unordered_set>


struct TraverseContext {
    PyObject *list;
    std::unordered_set<PyObject *> visited_objects;
};


static int
visit_callback(PyObject *object, void *arg) {
    auto *ctx = static_cast<TraverseContext *>(arg);

    if (ctx->visited_objects.count(object)) {
        return 0;
    }

    ctx->visited_objects.insert(object);

    if (PyList_Append(ctx->list, object) == -1) {
        return -1;
    }

    return 0;
}


static PyObject*
call_traverse(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "call_traverse() takes exactly one argument");
        return nullptr;
    }

    PyObject *obj = args[0];

    if (!Py_TYPE(obj)->tp_traverse) {
        PyErr_SetString(PyExc_TypeError, "Object does not support traversal");
        return nullptr;
    }

    PyObject *result_list = PyList_New(0);
    if (!result_list) {
        return nullptr;
    }

    TraverseContext ctx{result_list, {}};

    if (Py_TYPE(obj)->tp_traverse(obj, visit_callback, &ctx) == -1) {
        Py_DECREF(result_list);
        return nullptr;
    }

    return result_list;
}


static PyMethodDef TraverseMethods[] = {
    {"traverse", (PyCFunction)call_traverse, METH_FASTCALL,
     "Call tp_traverse on an object and return visited objects."},
    {nullptr, nullptr, 0, nullptr}
};


static struct PyModuleDef traverse_module = {
    PyModuleDef_HEAD_INIT,
    "traverse",
    "A module to call tp_traverse on Python objects.",
    -1,
    TraverseMethods
};


PyMODINIT_FUNC PyInit_traverse(void) {
    return PyModule_Create(&traverse_module);
}
