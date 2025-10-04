#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h> // For std::vector and std::unordered_map

#include "STPFCM.h"
#include "DatasetLoader.h"

namespace py = pybind11;

extern DatasetLoader dataset_loader;

PYBIND11_MODULE(_algofuzz, m) {
    m.doc() = "pybind11 plugin for Algofuzz C++ module"; // Optional module docstring

    py::class_<STPFCM>(m, "STPFCM")
        .def(py::init<int, int, float, float, float, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f,
             py::arg("p") = 2.0f,
             py::arg("kappa") = 1.0f,
             py::arg("w_prob") = 1.0f)
        .def("set_parameters", &STPFCM::setParameters,
             "Set parameters for the STPFCM model from a dictionary")
        .def("fit", &STPFCM::fit,
             "Fit the STPFCM model to the data",
             py::arg("X"))
        .def("set_centroids", &STPFCM::setCentroids,
             "Set initial centroids for the STPFCM model",
             py::arg("initial_centroids"))
        .def("is_trained", &STPFCM::isTrained,
             "Check if the model has been trained")
        .def("get_centroids", &STPFCM::getCentroids,
             "Get the cluster centroids")
        .def("get_member", &STPFCM::getMember,
             "Get the membership matrix")
        .def("get_alpha", &STPFCM::getAlpha,
             "Get the alpha vector")
        .def("get_eta", &STPFCM::getEta,
             "Get the eta matrix")
        .def("get_predicted_labels", &STPFCM::getPredictedLabels,
             "Get the predicted labels");

    py::class_<Dataset>(m, "Dataset")
        .def_readwrite("X", &Dataset::X)
        .def_readwrite("true_labels", &Dataset::true_labels)
        .def_readwrite("num_clusters", &Dataset::num_clusters)
        .def_readwrite("num_features", &Dataset::num_features)
        .def_readwrite("num_samples", &Dataset::num_samples);

    py::class_<DatasetLoader>(m, "DatasetLoader")
        .def("load_from_csv", &DatasetLoader::loadFromCSV,
             "Load dataset from a CSV file",
             py::arg("filename"), py::arg("normalize") = false)
        .def("load_from_numpy_array", &DatasetLoader::loadFromNumpyArray,
             "Load dataset from a NumPy array",
             py::arg("X"), py::arg("true_labels"), py::arg("num_clusters"))
        .def("get_dataset", &DatasetLoader::getDataset,
             "Get dataset by ID",
             py::arg("id"),
             py::return_value_policy::reference); // Return a reference to the existing object

    m.attr("dataset_loader") = py::cast(&dataset_loader, py::return_value_policy::reference);
}