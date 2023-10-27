#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <complex>
#include <functional>
#include <fstream>
 
using namespace Eigen;
using PrimeFunction = std::function<MatrixXcd(const double, const MatrixXcd&)>;

VectorXd maskA, maskB;
// std::ofstream paramsFile("params_cpp.txt");
// std::ofstream dataFile("data_cpp.txt");

void setMasks(const int Ns) 
{
    maskA = VectorXd::Zero(Ns);
    maskB = VectorXd::Zero(Ns);
    for (int i=0; i< Ns; i++) {
        if (i%2==0) maskA(i)=1;
        else maskB(i)=1;
    }
}

MatrixXd offDiagonal;
void setOffDiagonal(const int Ns, const double t1, const double t2)
{
    offDiagonal = MatrixXd::Zero(Ns,Ns);

    int N = (Ns-1)/2;

    double t_intra = t1, t_inter = t2;
    for (int i = 0; i < Ns-1; i++){ 
        if (i==N) { //swap t1/t2 when about to start second half of the bonds
            t_intra = t2; t_inter = t1;
        } 
        double t = (i%2==0? t_intra : t_inter);
        offDiagonal(i, i+1) = t;
        offDiagonal(i+1, i) = t;
        
    }
    
}


void rk4_step(const PrimeFunction& func,
     MatrixXcd& y, const double t, const double delta_t)
{
    MatrixXcd k1 = func(t, y);
    MatrixXcd k2 = func(t + 0.5*delta_t, y+ 0.5*k1*delta_t);
    MatrixXcd k3 = func(t + 0.5*delta_t, y+ 0.5*k2*delta_t);
    MatrixXcd k4 = func(t+delta_t, y+k3*delta_t);

    y = y + delta_t * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}

const MatrixXcd time_series(int N, double psi0, 
    double satGainA, double satGainB, double gammaA, double gammaB, 
    double time_end, double time_delta,
    double t1, double t2) {
    int Ns = 2*N+1;

    // if (paramsFile.is_open()) {
        
    //     paramsFile << N << std::endl;
    //     paramsFile << psi0 << std::endl;
    //     paramsFile << satGainA << std::endl;
    //     paramsFile << satGainB << std::endl;
    //     paramsFile << gammaA << std::endl;
    //     paramsFile << gammaB << std::endl;
    //     paramsFile << time_end << std::endl;
    //     paramsFile << time_delta << std::endl;
    //     paramsFile << t1 << std::endl;
    //     paramsFile << t2 << std::endl;
    //     paramsFile.close();
    // }

    

    MatrixXcd psi0_(Ns, 1);
    psi0_.setConstant(psi0);
    double time_start = 0.;
    VectorXd t_values = VectorXd::LinSpaced(static_cast<int>((time_end - time_start) / time_delta)+1, time_start, time_end);
    std::cout << "real delta t:" << t_values(1)-t_values(0)<< ","<< t_values(0)<< std::endl;
    std::cout << "num t: " << t_values.size() << std::endl;
    std::cout << "orig delta t: " << time_delta << std::endl;
    MatrixXcd psi_values(t_values.size(), Ns);

    MatrixXcd psi = psi0_;
    
    setMasks(Ns);
    setOffDiagonal(Ns, t1, t2);
    PrimeFunction psi_prime = [&](const double t, const MatrixXcd& psi) {
        MatrixXcd result(Ns,1);
        MatrixXcd satGainTerm = MatrixXcd::Ones(Ns,1).cwiseQuotient(MatrixXcd::Ones(Ns,1)+psi.cwiseAbs2());
        result = (satGainA*satGainTerm - MatrixXcd::Constant(Ns,1,gammaA)).cwiseProduct(psi).cwiseProduct(maskA);
        result += (satGainB*satGainTerm - MatrixXcd::Constant(Ns,1,gammaB)).cwiseProduct(psi).cwiseProduct(maskB);
        result -= std::complex<double>(0.,1.0)*offDiagonal*psi;
        // result = offDiagonal*psi/std::complex<double>(0.,1.0);
        return result;
    };
    
    for (int i = 0; i < t_values.size(); i++)
    {
        rk4_step(psi_prime, psi, t_values(i), time_delta);
        psi_values.row(i) = psi.transpose();
    }

    MatrixXcd t_psi_values(t_values.size(), Ns+1);
    t_psi_values << t_values , psi_values;

    // if (dataFile.is_open()) {
    //     for (int i = 0; i < t_psi_values.real().rows(); i++) {
    //         for (int j = 0; j < t_psi_values.real().cols(); j++) {
    //             dataFile << t_psi_values.real()(i, j) << " ";
    //         }
    //         dataFile << "\n";
    //     }
    //     dataFile.close();
    // }

    return t_psi_values;
}



PYBIND11_MODULE(ssh_1d, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("time_series", &time_series, "Time series of 1D SSH system with domain wall");
}

int main() {
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

    time_series(N, psi0, gA, gB, gamA, gamB, t_end, dt, t1, t2);
    return 0;
}

 