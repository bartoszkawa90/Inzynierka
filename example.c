// example.c

#include <Python.h>

static PyObject* example_add(PyObject* self, PyObject* args) {
    int a, b;

    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }

    int result = a + b;

    return Py_BuildValue("i", result);
}

static PyMethodDef example_methods[] = {
        {"add", example_add, METH_VARARGS, "Add two integers"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef example_module = {
        PyModuleDef_HEAD_INIT,
        "example",
        NULL,
        -1,
        example_methods
};

PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&example_module);
}
