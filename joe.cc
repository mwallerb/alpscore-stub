#include <iostream>
#include <random>

#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/autocorr.hpp>
#include <alps/alea/batch.hpp>

int main(int argc, char* argv[])
{

    long seed = std::random_device()();
    std::mt19937 gen(seed);
    std::normal_distribution<> dist (0.0, 1.0);

    alps::alea::mean_acc<double> mean (1);
    alps::alea::var_acc<double> var (1);
    alps::alea::autocorr_acc<double> autocorr (1);
    alps::alea::batch_acc<double> batch (1);

    int ncycles = 1000000;
    for (int i = 0; i < ncycles; ++i){
        double val = dist(gen);
        mean << val;
        var << val;
        autocorr << val;
        batch << val;
    }

    std::cout << "mean_acc\n";
    std::cout << "mean = " << mean.result().mean() << "\n";

    std::cout << "var_acc\n";
    std::cout << "mean = " << var.result().mean() << "\n";
    std::cout << "var = " << var.result().var() << "\n";

    std::cout << "autocorr_acc\n";
    alps::alea::autocorr_result<double> res = autocorr.result();
    for (size_t i = 0; i != res.nlevel(); ++i) {
        std::cout << " xbar=" << res.level(i).mean()
                  << " var=" << res.level(i).var()
                  << " cnt=" << res.level(i).count()
                  << " bs=" << res.level(i).batch_size()
                  << "\n";
    }

    std::cout << "mean = " << autocorr.result().mean() << "\n";
    std::cout << "var = " << autocorr.result().var() << "\n";
    std::cout << "tau = " << autocorr.result().tau() << "\n";


    std::cout << "batch_acc\n";
    std::cout << "mean = " << batch.result().mean() << "\n";
    std::cout << "var = " << batch.result().var() << "\n";

}
