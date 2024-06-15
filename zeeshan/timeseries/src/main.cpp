#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <iostream>
#include <complex>
#include <functional>
#include <fstream>
#include <vector>
#include <chrono>

using namespace Eigen;
using PrimeFunction = std::function<MatrixXcd(const double, const MatrixXcd&)>;
namespace py = pybind11;

using namespace std::complex_literals;

void rk4_step(const PrimeFunction& func, MatrixXcd& y, const double t, const double delta_t) {
    MatrixXcd k1 = func(t, y);
    MatrixXcd k2 = func(t + 0.5*delta_t, y+ 0.5*k1*delta_t);
    MatrixXcd k3 = func(t + 0.5*delta_t, y+ 0.5*k2*delta_t);
    MatrixXcd k4 = func(t+delta_t, y+k3*delta_t);

    y = y + delta_t * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}

class SSH_1D_satgain {//todo: later change this to a general "System" class, and unwanted features of system could be disabled by setting appropriate parameters to 0
public:
  int N; double psi0; double satGainA; double satGainB; double gammaA; double gammaB;
  double time_end; double time_delta;
  double t1; double t2;
  SSH_1D_satgain(int N, double psi0, 
    double satGainA, double satGainB, double gammaA, double gammaB, 
    double time_end, double time_delta,
    double t1, double t2):
    N{N}, psi0{psi0}, satGainA{satGainA}, satGainB{satGainB}, gammaA{gammaA},gammaB{gammaB},
    time_end{time_end}, time_delta{time_delta}, t1{t1}, t2{t2}
    { }

  const MatrixXcd simulate() {
    int Ns = 2*N+1;

    MatrixXcd psi0_(Ns, 1);
    psi0_.setConstant(psi0);
    double time_start = 0.;
    VectorXd t_values = VectorXd::LinSpaced(static_cast<int>((time_end - time_start) / time_delta)+1, time_start, time_end);
    MatrixXcd psi_values(t_values.size(), Ns);

    MatrixXcd psi = psi0_;

    MatrixXd offDiagonal = MatrixXd::Zero(Ns,Ns);

    double t_intra = t1, t_inter = t2;
    for (int i = 0; i < Ns-1; i++) {
      double t = (i%2==0? t_intra : t_inter);
      offDiagonal(i, i+1) = t;
      if (i==N) { t_intra = t2; t_inter = t1; }
      t = (i%2==0? t_intra : t_inter);
      offDiagonal(i+1, i) = t;
    }
    
    PrimeFunction psi_prime = [&](const double t, const MatrixXcd& psi) {
      MatrixXcd result(Ns,1);

      for (int i=0;i<Ns;i++) {
        double satGainTerm = 1./(1.+std::norm(psi(i,0)));
        double satGain, gamma;
        if (i%2==0) {
          satGain = satGainA; gamma = gammaA;
        } else {
          satGain = satGainB; gamma = gammaB;
        }
        result(i,0) = (satGain*satGainTerm-gamma)*psi(i,0);
      }
      result.noalias() -= 1.0i*offDiagonal*psi;
      return result;
    };

    for (int i = 0; i < t_values.size(); i++)
    {
      rk4_step(psi_prime, psi, t_values(i), time_delta);
      psi_values.row(i) = psi.transpose();
    }

    MatrixXcd t_psi_values(t_values.size(), Ns+1);
    t_psi_values << t_values , psi_values;

    return t_psi_values;
  }


};



PYBIND11_MODULE(timeseries, m) {

    py::class_<SSH_1D_satgain>(m, "SSH_1D_satgain")
        .def(py::init<int, double, double, double, double, double, double, double, double, double>(),
          py::arg("N"),py::arg("psi0"),py::arg("satGainA"),py::arg("satGainB"),py::arg("gammaA"),py::arg("gammaB"),
          py::arg("time_end"),py::arg("time_delta"),py::arg("t1"),py::arg("t2"))
        .def("simulate", &SSH_1D_satgain::simulate, 
          py::call_guard<py::gil_scoped_release>(), py::return_value_policy::reference_internal)
        .def("testfunc", &SSH_1D_satgain::testfunc)
        ;

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

  return 0;
}

