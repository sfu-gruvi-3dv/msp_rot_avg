#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "5point.h"

namespace py = pybind11;

py::tuple estimate_ransac_E(
        py::array_t<float> pts1,
        py::array_t<float> pts2,
        py::array_t<float> cam_K1,
        py::array_t<float> cam_K2,
        double ransac_threshold,
        int ransac_iterations
)
{
    // Check the types
    if(!py::isinstance<py::array_t<float_t>>(pts1))
        throw std::runtime_error("The pts1 array type should be 32-bit float.");
    if(!py::isinstance<py::array_t<float_t >>(pts2))
        throw std::runtime_error("The pts2 array type should be 32-bit float.");
    if(!py::isinstance<py::array_t<float_t >>(cam_K1))
        throw std::runtime_error("The cam_K1 array type should be 32-bit float.");
    if(!py::isinstance<py::array_t<float_t >>(cam_K2))
        throw std::runtime_error("The cam_K2 array type should be 32-bit float.");

    // Check the dimension
    if (pts1.shape(0) != pts2.shape(0))
        throw std::runtime_error("The number of matches in pts1 and pts2 should be same.");
    if (cam_K1.shape(0) != 3 && cam_K1.shape(1) != 3)
        throw std::runtime_error("The cam_K1 should be 3x3 matrix.");
    if (cam_K2.shape(0) != 3 && cam_K2.shape(1) != 3)
        throw std::runtime_error("The cam_K2 should be 3x3 matrix.");

    py::buffer_info pts1_buf = pts1.request();
    py::buffer_info pts2_buf = pts2.request();
    py::buffer_info K1_buf = cam_K1.request();
    py::buffer_info K2_buf = cam_K2.request();

    // buffer ptr
    float *pts1_buf_ptr = (float*)pts1_buf.ptr;
    float *pts2_buf_ptr = (float*)pts2_buf.ptr;
    float *K1_buf_ptr = (float*)K1_buf.ptr;
    float *K2_buf_ptr = (float*)K2_buf.ptr;

    // Run the 5-point algorithm
    int n_matches = pts1.shape(0);
    v2_t* pts1_t = new v2_t[n_matches];
    v2_t* pts2_t = new v2_t[n_matches];

    // Normalize the input
    double cx1 = K1_buf_ptr[2];
    double cy1 = K1_buf_ptr[5];
    double f1 = K1_buf_ptr[0];

    double cx2 = K2_buf_ptr[2];
    double cy2 = K2_buf_ptr[5];
    double f2 = K2_buf_ptr[0];

    for (int i=0; i< n_matches; i++)
    {
        float lx = pts1_buf_ptr[i*2];
        float ly = pts1_buf_ptr[i*2 + 1];
        float rx = pts2_buf_ptr[i*2];
        float ry = pts2_buf_ptr[i*2 + 1];

        // The coordinate used in bundler is different form normal ones
        pts1_t[i] = v2_new(lx-cx1, -ly+cy1);
        pts2_t[i] = v2_new(rx-cx2, -ry+cy2);
    }

    double K1[9], K2[9];
    K1[0] = K1[4] = f1; K1[8] = 1.0;
    K1[1] = K1[2] = K1[3] = K1[5] = K1[6] = K1[7] = 0;

    K2[0] = K2[4] = f2; K2[8] = 1.0;
    K2[1] = K2[2] = K2[3] = K2[5] = K2[6] = K2[7] = 0;

    double R0[9], t0[3], E[9];
    int num_inliers = compute_pose_ransac(n_matches, pts1_t, pts2_t,
                                          K1, K2,
                                          ransac_threshold, ransac_iterations, R0, t0, E); // 200 is the ransac round and you may change it.
    //change coordinate system
    R0[1] = -R0[1]; R0[2] = -R0[2];
    R0[3] = -R0[3]; R0[6] = -R0[6];
    t0[1] = -t0[1]; t0[2] = -t0[2];

    // gather output
    auto R_vec = py::array_t<float_t>(9);
    auto E_vec = py::array_t<float_t>(9);
    auto t_vec = py::array_t<float_t>(3);
    py::buffer_info R_vec_buf = R_vec.request(true);
    float *R_vec_buf_ptr = (float*)R_vec_buf.ptr;
    py::buffer_info E_vec_buf = E_vec.request(true);
    float *E_vec_buf_ptr = (float*)E_vec_buf.ptr;
    py::buffer_info t_vec_buf = t_vec.request(true);
    float *t_vec_buf_ptr = (float*)t_vec_buf.ptr;

    for (int j = 0; j < 9; ++j) {
        R_vec_buf_ptr[j] = (float) R0[j];
        E_vec_buf_ptr[j] = (float) E[j];
    }

    for (int j = 0; j < 3; ++j)
        t_vec_buf_ptr[j] = (float)t0[j];

    delete [] pts1_t;
    delete [] pts2_t;
    R_vec.resize({3, 3});
    return pybind11::make_tuple(R_vec, t_vec, num_inliers);
}

PYBIND11_MODULE(five_point_alg, m){
    m.doc() = "Python binding of 5-point algorithm.";
    m.def("estimate_ransac_E", &estimate_ransac_E, "Check the input array");
}
