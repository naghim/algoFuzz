#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h> // For std::vector and std::unordered_map

#include "BaseFCM.h"
#include "FCM.h"
#include "FCPlus1M.h"
#include "EtaFCM.h"
#include "FP3CM.h"
#include "GFPCM.h"
#include "PFCM.h"
#include "STPFCM.h"
#include "DatasetLoader.h"

namespace py = pybind11;

extern DatasetLoader dataset_loader;

PYBIND11_MODULE(_algofuzz, m) {
    m.doc() = "pybind11 plugin for Algofuzz C++ module"; // Optional module docstring

    py::class_<BaseFCM>(m, "BaseFCM")
        .def(py::init<int, int, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f)
        .def("set_parameters", &BaseFCM::setParameters,
             "Set parameters for the BaseFCM model from a dictionary")
        .def("set_centroids", &BaseFCM::setCentroids,
             "Set initial centroids for the BaseFCM model",
             py::arg("initial_centroids"))
        .def("is_trained", &BaseFCM::isTrained,
             "Check if the model has been trained")
        .def("get_centroids", &BaseFCM::getCentroids,
             "Get the cluster centroids")
        .def("get_member", &BaseFCM::getMember,
             "Get the membership matrix")
        .def("get_predicted_labels", &BaseFCM::getPredictedLabels,
             "Get the predicted labels");

    py::class_<FCM, BaseFCM>(m, "FCM")
        .def(py::init<int, int, float, float, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f,
             py::arg("kappa") = 1.0f,
             py::arg("noise") = 0.0f)
        .def("set_parameters", &FCM::setParameters,
             "Set parameters for the FCM model from a dictionary")
        .def("fit", &FCM::fit,
             "Fit the FCM model to the data",
             py::arg("X"));

    py::class_<STPFCM, BaseFCM>(m, "STPFCM")
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
        .def("get_alpha", &STPFCM::getAlpha,
             "Get the alpha vector")
        .def("get_eta", &STPFCM::getEta,
             "Get the eta matrix");

    py::class_<FCPlus1M, FCM>(m, "FCPlus1M")
        .def(py::init<int, int, float, float, float, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f,
             py::arg("kappa") = 1.0f,
             py::arg("eta") = 2.5f,
             py::arg("noise") = 0.0f)
        .def("set_parameters", &FCPlus1M::setParameters,
             "Set parameters for the FCPlus1M model from a dictionary")
        .def("fit", &FCPlus1M::fit,
             "Fit the FCPlus1M model to the data",
             py::arg("X"));

    py::class_<EtaFCM, FCM>(m, "EtaFCM")
        .def(py::init<int, int, float, float, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f,
             py::arg("kappa") = 1.0f,
             py::arg("noise") = 0.0f)
        .def("set_parameters", &EtaFCM::setParameters,
             "Set parameters for the EtaFCM model from a dictionary")
        .def("fit", &EtaFCM::fit,
             "Fit the EtaFCM model to the data",
             py::arg("X"))
        .def("get_eta", &EtaFCM::getEta,
             "Get the eta vector");

    py::class_<FP3CM, BaseFCM>(m, "FP3CM")
        .def(py::init<int, int, float, float, float, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f,
             py::arg("p") = 2.0f,
             py::arg("eta") = 0.1f,
             py::arg("noise") = 0.0f)
        .def("set_parameters", &FP3CM::setParameters,
             "Set parameters for the FP3CM model from a dictionary")
        .def("fit", &FP3CM::fit,
             "Fit the FP3CM model to the data",
             py::arg("X"));

    py::class_<GFPCM, BaseFCM>(m, "GFPCM")
        .def(py::init<int, int, float, float, float, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f,
             py::arg("p") = 2.0f,
             py::arg("w_prob") = 1.0f,
             py::arg("noise") = 0.0f)
        .def("set_parameters", &GFPCM::setParameters,
             "Set parameters for the GFPCM model from a dictionary")
        .def("fit", &GFPCM::fit,
             "Fit the GFPCM model to the data",
             py::arg("X"));

    py::class_<PFCM, BaseFCM>(m, "PFCM")
        .def(py::init<int, int, float, int, float, float, float, float>(),
             py::arg("num_clusters"),
             py::arg("max_iter"),
             py::arg("m") = 2.0f,
             py::arg("preprocess_iter") = 15,
             py::arg("p") = 2.0f,
             py::arg("w_pos") = 1.0f,
             py::arg("w_prob") = 1.0f,
             py::arg("noise") = 0.0f)
        .def("set_parameters", &PFCM::setParameters,
             "Set parameters for the PFCM model from a dictionary")
        .def("fit", &PFCM::fit,
             "Fit the PFCM model to the data",
             py::arg("X"));

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