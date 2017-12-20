#include <iostream>

#include <alps/alea.hpp>
#include <alps/mc/random01.hpp>


struct my_estimator
    : public alps::alea::computed<double>
{
    my_estimator(const alps::random01 &rng = alps::random01())
        : rng_(rng), current_(rng_())
    { }

    void next() { current_ = rng_(); }

    // implementations

    size_t size() const { return 1; }

    void add_to(alps::alea::sink<double> out) const
    {
        out.data()[0] += current_;
        out.data()[1] += 2 * current_;
    }

private:
    alps::random01 rng_;
    double current_;
};

int main(int argc, char** argv)
{
    alps::mpi::environment env(argc, argv, false);

    Eigen::Vector2d x;
    alps::alea::var_acc<double> acc(2);
    my_estimator estimator;
    for (size_t i = 0; i != 1000; ++i) {
        //estimator.next();
        x << i, i+1;
        acc << x;
    }

    alps::alea::var_result<double> res = acc.result();
    res.reduce(alps::alea::mpi_reducer());

    if (res.valid()) {
        std::cout << "\nMean: " << res.mean()
                  << "\nVariance: " << res.var() << std::endl;

        alps::alea::mean_result<double> res2 = alps::alea::transform(
                        alps::alea::no_prop(),
                        alps::alea::make_transformer<double>(
                            [] (double x, double y) -> double { return x * y; }
                            ),
                        res);

        std::cout << "\nMean: " << res2.mean() << std::endl;
    }
}

