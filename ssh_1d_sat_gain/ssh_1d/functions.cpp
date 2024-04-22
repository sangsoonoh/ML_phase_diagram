#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <iostream>
#include <complex>
#include <functional>
#include <fstream>
#include <vector>
#include <chrono>
#include <BS_thread_pool.hpp>

using namespace Eigen;
using PrimeFunction = std::function<MatrixXcd(const double, const MatrixXcd&)>;
namespace py = pybind11;

using namespace std::complex_literals;

class TimeSeriesParams {
public:
    int N; double psi0; double satGainA; double satGainB; double gammaA; double gammaB;
    double time_end; double time_delta;
    double t1; double t2;
    TimeSeriesParams(int N, double psi0, 
        double satGainA, double satGainB, double gammaA, double gammaB, 
        double time_end, double time_delta,
        double t1, double t2):
        N{N}, psi0{psi0}, satGainA{satGainA}, satGainB{satGainB}, gammaA{gammaA},gammaB{gammaB},
        time_end{time_end}, time_delta{time_delta}, t1{t1}, t2{t2}
        { }
};


void rk4_step(const PrimeFunction& func,
     MatrixXcd& y, const double t, const double delta_t)
{
    MatrixXcd k1 = func(t, y);
    MatrixXcd k2 = func(t + 0.5*delta_t, y+ 0.5*k1*delta_t);
    MatrixXcd k3 = func(t + 0.5*delta_t, y+ 0.5*k2*delta_t);
    MatrixXcd k4 = func(t+delta_t, y+k3*delta_t);

    y = y + delta_t * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}

const MatrixXcd time_series(const TimeSeriesParams& params) {
    TimeSeriesParams p = params;
    int Ns = 2*p.N+1;

    MatrixXcd psi0_(Ns, 1);
    psi0_.setConstant(p.psi0);
    double time_start = 0.;
    VectorXd t_values = VectorXd::LinSpaced(static_cast<int>((p.time_end - time_start) / p.time_delta)+1, time_start, p.time_end);
    MatrixXcd psi_values(t_values.size(), Ns);

    MatrixXcd psi = psi0_;


    MatrixXd offDiagonal = MatrixXd::Zero(Ns,Ns);

    double t_intra = p.t1, t_inter = p.t2;
    for (int i = 0; i < Ns-1; i++) {
        double t = (i%2==0? t_intra : t_inter);
        offDiagonal(i, i+1) = t;
        if (i==p.N) { t_intra = p.t2; t_inter = p.t1; }
        t = (i%2==0? t_intra : t_inter);
        offDiagonal(i+1, i) = t;
    }
    
    PrimeFunction psi_prime = [&](const double t, const MatrixXcd& psi) {
        MatrixXcd result(Ns,1);

        for (int i=0;i<Ns;i++) {
            double satGainTerm = 1./(1.+std::norm(psi(i,0)));
            double satGain, gamma;
            if (i%2==0) {
                satGain = p.satGainA; gamma = p.gammaA;
            } else {
                satGain = p.satGainB; gamma = p.gammaB;
            }
            result(i,0) = (satGain*satGainTerm-gamma)*psi(i,0);
        }
        result.noalias() -= 1.0i*offDiagonal*psi;
        return result;
    };

    for (int i = 0; i < t_values.size(); i++)
    {
        rk4_step(psi_prime, psi, t_values(i), p.time_delta);
        psi_values.row(i) = psi.transpose();
    }

    MatrixXcd t_psi_values(t_values.size(), Ns+1);
    t_psi_values << t_values , psi_values;

    return t_psi_values;
}

double time_series_oscillating_factor(const TimeSeriesParams& params) {
    int powOfTwo = (int) pow(2.0,16);
    const MatrixXcd & time_series_mat = time_series(params);


    int end_time_index = time_series_mat.rows()-1;
    int start_time_index = end_time_index-powOfTwo+1;

    FFT<double> fft;

    MatrixXcd selected_psi = time_series_mat.block(start_time_index,1,end_time_index-start_time_index+1, time_series_mat.cols()-1);
    MatrixXcd psi_fft(selected_psi.rows(),selected_psi.cols());
    for (int i=0; i< selected_psi.cols(); i++)
    {
        VectorXcd dst(powOfTwo, 1);
        fft.fwd(dst, selected_psi.col(i));
        psi_fft.col(i) = dst;
    }
    psi_fft.topRows(10).setZero();


    double of = psi_fft.cwiseAbs2().rowwise().sum().maxCoeff();
    
    return of;
}

VectorXd calc_phase_diagram_parallel(const std::vector<TimeSeriesParams>& paramss)
{
    VectorXd results(paramss.size());
    results.setZero();
    BS::thread_pool pool;
    auto loop = [&paramss, &results](int a, int b){
        for (int i=a; i<b; i++) {
            results(i) = time_series_oscillating_factor(paramss.at(i));
        }
        
    };
    pool.parallelize_loop(paramss.size(), loop).wait();
    std::cout <<results(2)<<std::endl;
    return results;
}


PYBIND11_MODULE(ssh_1d, m) {
    m.doc() = "Module for time simulating 1D SSH system with domain-wall in the middle";

    py::class_<TimeSeriesParams>(m, "TimeSeriesParams")
        .def(py::init<int, double, double, double, double, double, double, double, double, double>(),
        py::arg("N"),py::arg("psi0"),py::arg("satGainA"),py::arg("satGainB"),py::arg("gammaA"),py::arg("gammaB"),
        py::arg("time_end"),py::arg("time_delta"),py::arg("t1"),py::arg("t2"));

    m.def("time_series", &time_series, "Time series of 1D SSH system with domain wall", py::call_guard<py::gil_scoped_release>(),
        py::return_value_policy::reference_internal,
        py::arg("params"));

    m.def("time_series_oscillating_factor", &time_series_oscillating_factor, "Oscillating factor of time series of 1D SSH system with domain wall", 
        py::call_guard<py::gil_scoped_release>(),
        py::arg("params"));

    m.def("calc_phase_diagram_parallel", &calc_phase_diagram_parallel);
}

int main() {
    //main function for testing c++ code directly

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    int N = 10;
    double psi0 = 0.01;
    double t_end = 1200;
    double dt = 0.01;
    double gamA = 0.48;
    double gamB = gamA;
    double gA = 0.06 + gamA;
    double gB = 0;
    double t1 = 1;
    double t2 = 0.7;

    TimeSeriesParams params(N, psi0, gA, gB, gamA, gamB, t_end, dt, t1, t2);
    std::vector<TimeSeriesParams> paramss;
    for (int i=0; i<2000; i++) {
        paramss.push_back(TimeSeriesParams(params));
    }

    auto chrono_t1 = high_resolution_clock::now(); // ***CHRONO START***
    VectorXd ofs = calc_phase_diagram_parallel(paramss);
    auto chrono_t2 = high_resolution_clock::now();// ***CHRONO END***
    duration<double, std::milli> ms_double = chrono_t2 - chrono_t1;
    std::cout << "chrono: "<< ms_double.count() << "ms\n";

    return 0;
}

