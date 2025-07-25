#include "RachfordRice.H"
#include <cmath>
#include <algorithm> // for std::max
#include "error.H"   // for FatalErrorInFunction, WarningInFunction

bool rachfordRiceSolve(
    const Foam::List<Foam::scalar>& K_values,
    const Foam::List<Foam::scalar>& Z,
    Foam::scalar& beta,
    Foam::scalar tol,
    Foam::scalar tolF,
    int maxIter,
    Foam::scalar relaxation)
{
    using namespace Foam;

    Foam::scalar betaLower = 0.0, betaUpper = 1.0;

    auto f = [&](Foam::scalar b) {
        Foam::scalar sum = 0.0;
        forAll(Z, i)
        {
            Foam::scalar denom = std::max(1.0 + b * (K_values[i]-1.0), Foam::SMALL);
            sum += Z[i] * (K_values[i]-1.0) / denom;
        }
        return sum;
    };

    Foam::scalar fLow = f(betaLower), fUp = f(betaUpper);
    if (fLow * fUp > 0)
    {
        return false;
    }

    for (int iter = 0; iter < maxIter; ++iter)
    {
        Foam::scalar fb = 0.0, dfb = 0.0;
        forAll(Z, i)
        {
            Foam::scalar denom = std::max(1.0 + beta * (K_values[i]-1.0), Foam::SMALL);
            Foam::scalar num = K_values[i]-1.0;
            fb  += Z[i] * num / denom;
            dfb -= Z[i] * num*num / (denom*denom);
        }
        Foam::scalar betaNew = beta - relaxation * fb/dfb;
        if (betaNew < betaLower || betaNew > betaUpper || !std::isfinite(betaNew))
            betaNew = 0.5 * (betaLower + betaUpper);
        Foam::scalar fNew = f(betaNew);
        if (fNew > 0) betaLower = std::max(betaNew, 0.0);
        else          betaUpper = std::min(betaNew, 1.0);
        if (std::abs(betaNew - beta) < tol || std::abs(fNew) < tolF)
        {
            beta = betaNew;
            return true;
        }
        beta = betaNew;
    }
    return false;
}
